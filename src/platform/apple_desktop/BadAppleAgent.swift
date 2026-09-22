import Foundation

public enum BadAppleAgentTaskStatus: String, Codable, Sendable {
    case queued
    case running
    case paused
    case completed
    case failed
    case cancelled

    public var isTerminal: Bool {
        self == .completed || self == .failed || self == .cancelled
    }
}

public struct BadAppleAgentPlannedStep: Codable, Equatable, Sendable {
    public let instruction: String

    public init(instruction: String) {
        self.instruction = instruction
    }
}

public struct BadAppleAgentAction: Codable, Equatable, Sendable {
    public let thought: String
    public let tool: String?
    public let arguments: [String: String]
    public let finish: String?

    public init(
        thought: String = "",
        tool: String? = nil,
        arguments: [String: String] = [:],
        finish: String? = nil
    ) {
        self.thought = thought
        self.tool = tool
        self.arguments = arguments
        self.finish = finish
    }

    public static func call(
        tool: String,
        arguments: [String: String] = [:],
        thought: String = ""
    ) -> BadAppleAgentAction {
        BadAppleAgentAction(thought: thought, tool: tool, arguments: arguments)
    }

    public static func finish(_ text: String, thought: String = "") -> BadAppleAgentAction {
        BadAppleAgentAction(thought: thought, finish: text)
    }
}

public struct BadAppleAgentStep: Codable, Equatable, Sendable {
    public let index: Int
    public let instruction: String
    public let thought: String
    public let tool: String
    public let arguments: [String: String]
    public let result: String
    public let error: String
    public let startedAt: Date
    public let completedAt: Date

    public init(
        index: Int,
        instruction: String,
        thought: String,
        tool: String,
        arguments: [String: String],
        result: String,
        error: String = "",
        startedAt: Date,
        completedAt: Date
    ) {
        self.index = index
        self.instruction = instruction
        self.thought = thought
        self.tool = tool
        self.arguments = arguments
        self.result = result
        self.error = error
        self.startedAt = startedAt
        self.completedAt = completedAt
    }
}

public struct BadAppleAgentTask: Codable, Equatable, Sendable {
    public let id: String
    public let goal: String
    public var status: BadAppleAgentTaskStatus
    public let maxSteps: Int
    public let createdAt: Date
    public var updatedAt: Date
    public var plan: [BadAppleAgentPlannedStep]
    public var steps: [BadAppleAgentStep]
    public var summary: String
    public var error: String

    public init(
        id: String,
        goal: String,
        status: BadAppleAgentTaskStatus,
        maxSteps: Int,
        createdAt: Date,
        updatedAt: Date,
        plan: [BadAppleAgentPlannedStep] = [],
        steps: [BadAppleAgentStep] = [],
        summary: String = "",
        error: String = ""
    ) {
        self.id = id
        self.goal = goal
        self.status = status
        self.maxSteps = maxSteps
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.plan = plan
        self.steps = steps
        self.summary = summary
        self.error = error
    }
}

public struct BadAppleAgentGenerationContext: Codable, Equatable, Sendable {
    public let taskID: String
    public let goal: String
    public let plannedStep: BadAppleAgentPlannedStep
    public let stepIndex: Int
    public let maxSteps: Int
    public let completedSteps: [BadAppleAgentStep]

    public init(
        taskID: String,
        goal: String,
        plannedStep: BadAppleAgentPlannedStep,
        stepIndex: Int,
        maxSteps: Int,
        completedSteps: [BadAppleAgentStep]
    ) {
        self.taskID = taskID
        self.goal = goal
        self.plannedStep = plannedStep
        self.stepIndex = stepIndex
        self.maxSteps = maxSteps
        self.completedSteps = completedSteps
    }
}

public enum BadAppleAgentError: Error, LocalizedError, Equatable {
    case emptyGoal
    case persistence(String)

    public var errorDescription: String? {
        switch self {
        case .emptyGoal:
            return "The agent goal cannot be empty."
        case .persistence(let message):
            return "Agent task persistence failed: \(message)"
        }
    }
}

/// A persistent, Swift-native plan-and-execute task manager.
///
/// The executor must be the application's policy-aware executor (for example,
/// `BadAppleToolExecutor.executeTool`). This actor never marks calls as approved
/// and treats approval-required or policy responses as reasons to pause a task.
public actor BadAppleAgent {
    public typealias Planner = @Sendable (
        _ goal: String,
        _ maximumSteps: Int
    ) async throws -> [BadAppleAgentPlannedStep]

    public typealias Generator = @Sendable (
        _ context: BadAppleAgentGenerationContext
    ) async throws -> BadAppleAgentAction

    public typealias ToolExecutor = @Sendable (
        _ tool: String,
        _ arguments: [String: String],
        _ purpose: String
    ) async throws -> String

    public static let minimumStepCount = 1
    public static let maximumStepCount = 50

    private struct RunningTask {
        let token: UUID
        let task: Task<Void, Never>
    }

    private let storageDirectory: URL
    private let planner: Planner
    private let generator: Generator
    private let executor: ToolExecutor
    private let fileManager: FileManager
    private var tasks: [String: BadAppleAgentTask]
    private var runningTasks: [String: RunningTask] = [:]

    public init(
        storageDirectory: URL = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".bad_apple", isDirectory: true)
            .appendingPathComponent("agent_tasks", isDirectory: true),
        planner: @escaping Planner,
        generator: @escaping Generator,
        executor: @escaping ToolExecutor
    ) throws {
        let fileManager = FileManager.default
        do {
            try fileManager.createDirectory(
                at: storageDirectory,
                withIntermediateDirectories: true,
                attributes: nil
            )
            self.tasks = try Self.loadTasks(from: storageDirectory, fileManager: fileManager)
        } catch {
            throw BadAppleAgentError.persistence(error.localizedDescription)
        }
        self.storageDirectory = storageDirectory
        self.planner = planner
        self.generator = generator
        self.executor = executor
        self.fileManager = fileManager
    }

    @discardableResult
    public func submit(goal: String, maxSteps: Int = 10) throws -> BadAppleAgentTask {
        let normalizedGoal = goal.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalizedGoal.isEmpty else { throw BadAppleAgentError.emptyGoal }

        let now = Date()
        let boundedSteps = min(max(maxSteps, Self.minimumStepCount), Self.maximumStepCount)
        let task = BadAppleAgentTask(
            id: UUID().uuidString.lowercased(),
            goal: normalizedGoal,
            status: .queued,
            maxSteps: boundedSteps,
            createdAt: now,
            updatedAt: now
        )
        tasks[task.id] = task
        do {
            try persist(task)
        } catch {
            tasks.removeValue(forKey: task.id)
            throw error
        }
        start(taskID: task.id)
        return task
    }

    public func list() -> [BadAppleAgentTask] {
        tasks.values.sorted { $0.createdAt > $1.createdAt }
    }

    public func get(taskID: String) -> BadAppleAgentTask? {
        tasks[taskID]
    }

    @discardableResult
    public func pause(taskID: String) throws -> Bool {
        guard var task = tasks[taskID], task.status == .running || task.status == .queued else {
            return false
        }
        task.status = .paused
        task.updatedAt = Date()
        tasks[taskID] = task
        try persist(task)
        runningTasks[taskID]?.task.cancel()
        return true
    }

    @discardableResult
    public func resume(taskID: String) throws -> Bool {
        guard var task = tasks[taskID], task.status == .paused else { return false }
        task.status = .queued
        task.error = ""
        task.updatedAt = Date()
        tasks[taskID] = task
        try persist(task)
        if let previousRun = runningTasks[taskID]?.task {
            Task { [weak self] in
                await previousRun.value
                guard let self else { return }
                await self.startIfQueued(taskID: taskID)
            }
        } else {
            start(taskID: taskID)
        }
        return true
    }

    @discardableResult
    public func cancel(taskID: String) throws -> Bool {
        guard var task = tasks[taskID], !task.status.isTerminal else { return false }
        task.status = .cancelled
        task.error = ""
        task.updatedAt = Date()
        tasks[taskID] = task
        try persist(task)
        runningTasks[taskID]?.task.cancel()
        return true
    }

    private func startIfQueued(taskID: String) {
        guard tasks[taskID]?.status == .queued, runningTasks[taskID] == nil else { return }
        start(taskID: taskID)
    }

    private func start(taskID: String) {
        let token = UUID()
        let handle = Task { [weak self] in
            guard let self else { return }
            await self.execute(taskID: taskID, runnerToken: token)
        }
        runningTasks[taskID] = RunningTask(token: token, task: handle)
    }

    private func execute(taskID: String, runnerToken: UUID) async {
        defer { finishRunner(taskID: taskID, token: runnerToken) }
        guard var task = tasks[taskID], task.status == .queued else { return }

        do {
            task.status = .running
            task.updatedAt = Date()
            tasks[taskID] = task
            try persist(task)

            if task.plan.isEmpty {
                let proposedPlan = try await planner(task.goal, task.maxSteps)
                guard canContinue(taskID: taskID) else { return }
                let normalizedPlan = proposedPlan
                    .filter { !$0.instruction.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
                    .prefix(task.maxSteps)
                guard !normalizedPlan.isEmpty else {
                    try fail(taskID: taskID, message: "Planner returned no executable steps.")
                    return
                }
                guard var current = tasks[taskID] else { return }
                current.plan = Array(normalizedPlan)
                current.updatedAt = Date()
                tasks[taskID] = current
                try persist(current)
            }

            while canContinue(taskID: taskID) {
                guard let current = tasks[taskID] else { return }
                let index = current.steps.count
                guard index < current.plan.count else {
                    try complete(taskID: taskID, summary: current.steps.last?.result ?? "Task completed.")
                    return
                }
                guard index < current.maxSteps else {
                    try fail(taskID: taskID, message: "Reached step limit (\(current.maxSteps)).")
                    return
                }

                let context = BadAppleAgentGenerationContext(
                    taskID: current.id,
                    goal: current.goal,
                    plannedStep: current.plan[index],
                    stepIndex: index,
                    maxSteps: current.maxSteps,
                    completedSteps: current.steps
                )
                let action = try await generator(context)
                guard canContinue(taskID: taskID) else { return }

                if let finish = action.finish {
                    let summary = finish.trimmingCharacters(in: .whitespacesAndNewlines)
                    try complete(taskID: taskID, summary: summary.isEmpty ? "Task completed." : summary)
                    return
                }

                guard let tool = action.tool?.trimmingCharacters(in: .whitespacesAndNewlines),
                      !tool.isEmpty else {
                    try fail(taskID: taskID, message: "Generator did not choose a tool or finish the task.")
                    return
                }

                let startedAt = Date()
                let result = try await executor(tool, action.arguments, "agent task: \(current.goal)")
                guard runningTasks[taskID]?.token == runnerToken,
                      let postExecutionTask = tasks[taskID],
                      postExecutionTask.status == .running
                        || postExecutionTask.status == .paused
                        || postExecutionTask.status == .queued else { return }
                let recordedResult = String(result.prefix(16_000))
                let interrupted = Self.isPolicyOrApprovalResponse(recordedResult)
                let step = BadAppleAgentStep(
                    index: index,
                    instruction: context.plannedStep.instruction,
                    thought: action.thought,
                    tool: tool,
                    arguments: action.arguments,
                    result: recordedResult,
                    error: interrupted ? recordedResult : "",
                    startedAt: startedAt,
                    completedAt: Date()
                )
                guard var updated = tasks[taskID] else { return }
                updated.steps.append(step)
                updated.updatedAt = Date()
                if interrupted {
                    updated.status = .paused
                    updated.error = recordedResult
                }
                tasks[taskID] = updated
                try persist(updated)
                if interrupted { return }
            }
        } catch is CancellationError {
            return
        } catch {
            guard let current = tasks[taskID], !current.status.isTerminal, current.status != .paused else {
                return
            }
            try? fail(taskID: taskID, message: error.localizedDescription)
        }
    }

    private func canContinue(taskID: String) -> Bool {
        guard !Task.isCancelled, let task = tasks[taskID] else { return false }
        return task.status == .running
    }

    private func complete(taskID: String, summary: String) throws {
        guard var task = tasks[taskID], task.status == .running else { return }
        task.status = .completed
        task.summary = summary
        task.error = ""
        task.updatedAt = Date()
        tasks[taskID] = task
        try persist(task)
        BadAppleNotify.push(
            kind: "task_done",
            title: "Task finished",
            body: "\(task.goal.prefix(140)) — \(summary.prefix(140))",
            voice: false
        )
    }

    private func fail(taskID: String, message: String) throws {
        guard var task = tasks[taskID], !task.status.isTerminal else { return }
        task.status = .failed
        task.error = String(message.prefix(16_000))
        task.updatedAt = Date()
        tasks[taskID] = task
        try persist(task)
        BadAppleNotify.push(
            kind: "task_failed",
            title: "Task failed",
            body: "\(task.goal.prefix(140)) — \(message.prefix(140))",
            voice: true
        )
    }

    private func finishRunner(taskID: String, token: UUID) {
        if runningTasks[taskID]?.token == token {
            runningTasks.removeValue(forKey: taskID)
        }
    }

    private func persist(_ task: BadAppleAgentTask) throws {
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let data = try encoder.encode(task)
            try data.write(to: path(for: task.id), options: .atomic)
        } catch {
            throw BadAppleAgentError.persistence(error.localizedDescription)
        }
    }

    private func path(for taskID: String) -> URL {
        storageDirectory.appendingPathComponent(taskID, isDirectory: false).appendingPathExtension("json")
    }

    private static func loadTasks(
        from directory: URL,
        fileManager: FileManager
    ) throws -> [String: BadAppleAgentTask] {
        let urls = try fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        )
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        var loaded: [String: BadAppleAgentTask] = [:]
        for url in urls where url.pathExtension.lowercased() == "json" {
            let data = try Data(contentsOf: url)
            var task = try decoder.decode(BadAppleAgentTask.self, from: data)
            guard isSafeTaskID(task.id), url.deletingPathExtension().lastPathComponent == task.id else {
                continue
            }
            if task.status == .running || task.status == .queued {
                task.status = .paused
                task.error = "Execution was interrupted before this process started. Resume to continue."
                task.updatedAt = Date()
            }
            loaded[task.id] = task
        }
        return loaded
    }

    private static func isSafeTaskID(_ value: String) -> Bool {
        guard !value.isEmpty, value.count <= 64 else { return false }
        return value.unicodeScalars.allSatisfy {
            CharacterSet.alphanumerics.contains($0) || $0 == "-" || $0 == "_"
        }
    }

    private static func isPolicyOrApprovalResponse(_ result: String) -> Bool {
        let normalized = result.lowercased()
        return normalized.hasPrefix("policy:")
            || normalized.contains("approval required")
            || normalized.contains("needs your approval")
            || normalized.contains("requires approval")
    }
}
