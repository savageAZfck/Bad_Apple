import Foundation

private struct TestFailure: Error, CustomStringConvertible {
    let description: String
}

private actor MockRecorder {
    private(set) var executions: [String] = []

    func record(_ tool: String) {
        executions.append(tool)
    }

    func snapshot() -> [String] {
        executions
    }
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() {
        throw TestFailure(description: message)
    }
}

private func waitForStatus(
    _ expected: BadAppleAgentTaskStatus,
    taskID: String,
    agent: BadAppleAgent,
    timeoutNanoseconds: UInt64 = 3_000_000_000
) async throws -> BadAppleAgentTask {
    let clock = ContinuousClock()
    let deadline = clock.now.advanced(by: .nanoseconds(Int64(timeoutNanoseconds)))
    while clock.now < deadline {
        if let task = await agent.get(taskID: taskID), task.status == expected {
            return task
        }
        try await Task.sleep(nanoseconds: 10_000_000)
    }
    let actual = await agent.get(taskID: taskID)?.status.rawValue ?? "missing"
    throw TestFailure(description: "Timed out waiting for \(expected.rawValue); current status is \(actual)")
}

@main
private enum BadAppleAgentExecutableTests {
    static func main() async {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("badapple-agent-tests-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }

        do {
            let recorder = MockRecorder()
            let planner: BadAppleAgent.Planner = { goal, _ in
                if goal == "pause and resume" {
                    try await Task.sleep(nanoseconds: 250_000_000)
                    return [BadAppleAgentPlannedStep(instruction: "resume step")]
                }
                if goal == "cancel this task" {
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                    return [BadAppleAgentPlannedStep(instruction: "must not execute")]
                }
                if goal == "two step task" {
                    return [
                        BadAppleAgentPlannedStep(instruction: "first step"),
                        BadAppleAgentPlannedStep(instruction: "second step"),
                    ]
                }
                return [BadAppleAgentPlannedStep(instruction: "single step")]
            }
            let generator: BadAppleAgent.Generator = { context in
                .call(
                    tool: context.goal == "approval gated" ? "destructive_mock" : "mock_tool_\(context.stepIndex + 1)",
                    arguments: ["instruction": context.plannedStep.instruction],
                    thought: "execute planned step"
                )
            }
            let executor: BadAppleAgent.ToolExecutor = { tool, arguments, purpose in
                await recorder.record(tool)
                if tool == "destructive_mock" {
                    return "Approval required before I can run destructive_mock."
                }
                return "\(tool) completed for \(purpose): \(arguments["instruction"] ?? "")"
            }

            let agent = try BadAppleAgent(
                storageDirectory: root,
                planner: planner,
                generator: generator,
                executor: executor
            )

            let pauseTask = try await agent.submit(goal: "pause and resume", maxSteps: 0)
            try require(pauseTask.status == .queued, "submit must return a queued task")
            try require(pauseTask.maxSteps == 1, "maxSteps must be bounded to at least one")
            _ = try await waitForStatus(.running, taskID: pauseTask.id, agent: agent)
            let didPause = try await agent.pause(taskID: pauseTask.id)
            try require(didPause, "running task should pause")
            _ = try await waitForStatus(.paused, taskID: pauseTask.id, agent: agent)
            let didResume = try await agent.resume(taskID: pauseTask.id)
            try require(didResume, "paused task should resume")
            let resumed = try await waitForStatus(.completed, taskID: pauseTask.id, agent: agent)
            try require(resumed.steps.count == 1, "resumed task should execute one step")

            let cancelTask = try await agent.submit(goal: "cancel this task")
            _ = try await waitForStatus(.running, taskID: cancelTask.id, agent: agent)
            let didCancel = try await agent.cancel(taskID: cancelTask.id)
            try require(didCancel, "running task should cancel")
            let cancelled = try await waitForStatus(.cancelled, taskID: cancelTask.id, agent: agent)
            try require(cancelled.steps.isEmpty, "cancelled task must not execute a tool")
            let resumedCancelledTask = try await agent.resume(taskID: cancelTask.id)
            try require(!resumedCancelledTask, "cancelled task must not resume")

            let twoStepTask = try await agent.submit(goal: "two step task", maxSteps: 2)
            let completed = try await waitForStatus(.completed, taskID: twoStepTask.id, agent: agent)
            try require(completed.plan.count == 2, "planner output should be persisted")
            try require(completed.steps.count == 2, "two planned tools should execute")
            try require(completed.steps.map(\.tool) == ["mock_tool_1", "mock_tool_2"], "steps ran out of order")
            try require(completed.summary.contains("mock_tool_2 completed"), "last result should summarize completion")

            let approvalTask = try await agent.submit(goal: "approval gated")
            let approvalPaused = try await waitForStatus(.paused, taskID: approvalTask.id, agent: agent)
            try require(approvalPaused.steps.count == 1, "approval response should be recorded")
            try require(approvalPaused.error.contains("Approval required"), "approval response should pause without bypassing policy")

            let listed = await agent.list()
            try require(listed.count == 4, "list should return all submitted tasks")
            let fetched = await agent.get(taskID: twoStepTask.id)
            try require(fetched != nil, "get should find submitted task")

            let persistedFiles = try FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
                .filter { $0.pathExtension == "json" }
            try require(persistedFiles.count == 4, "each task should have a persisted JSON file")

            let reloadedAgent = try BadAppleAgent(
                storageDirectory: root,
                planner: planner,
                generator: generator,
                executor: executor
            )
            guard let reloaded = await reloadedAgent.get(taskID: twoStepTask.id) else {
                throw TestFailure(description: "persisted task was not reloaded")
            }
            try require(reloaded.goal == completed.goal, "reloaded goal should match")
            try require(reloaded.status == completed.status, "reloaded status should match")
            try require(reloaded.plan == completed.plan, "reloaded plan should match")
            try require(reloaded.steps.map(\.tool) == completed.steps.map(\.tool), "reloaded steps should match")
            try require(reloaded.summary == completed.summary, "reloaded summary should match")
            let reloadedList = await reloadedAgent.list()
            try require(reloadedList.count == 4, "reloaded list should contain every task")

            let calls = await recorder.snapshot()
            try require(
                calls == ["mock_tool_1", "mock_tool_1", "mock_tool_2", "destructive_mock"],
                "unexpected executor calls: \(calls)"
            )

            print("BadAppleAgent tests passed: submit, pause/resume, cancel, persistence, two-step execution")
        } catch {
            fputs("BadAppleAgent tests failed: \(error)\n", stderr)
            exit(1)
        }
    }
}
