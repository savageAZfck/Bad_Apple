// BadAppleWorkspaceWatcher — native FSEvents-based workspace watcher.
//
// Wires the workspace into the daemon so file changes under the active
// workspace trigger `index_documents` automatically, keeping the working memory
// and RAG index fresh without user intervention.

import Foundation
import CoreServices

private func watcherLog(_ message: String) {
    NSLog("[workspace] %@", message)
}

final class BadAppleWorkspaceWatcher {
    private var stream: FSEventStreamRef?
    private let path: String
    private var pending: Set<String> = []
    private var flushWorkItem: DispatchWorkItem?
    private let lock = NSLock()
    private var indexCallback: ((String) -> Void)?
    private var running = false

    init(path: String) {
        self.path = path
    }

    /// Start watching the workspace and call `onChange` with the workspace
    /// path after a coalescing delay. FSEvents and flush debounce run on the
    /// main dispatch queue so this works under `dispatchMain()` without a
    /// `RunLoop`.
    func start(onChange: @escaping (String) -> Void) {
        DispatchQueue.main.async { [weak self] in
            self?._start(onChange: onChange)
        }
    }

    private func _start(onChange: @escaping (String) -> Void) {
        _stop()
        indexCallback = onChange

        var context = FSEventStreamContext(
            version: 0,
            info: Unmanaged.passUnretained(self).toOpaque(),
            retain: nil,
            release: nil,
            copyDescription: nil
        )

        let paths = [path] as CFArray
        let callback: FSEventStreamCallback = { (streamRef, clientCallBackInfo, numEvents, eventPaths, eventFlags, eventIds) in
            guard let clientCallBackInfo else { return }
            let watcher = Unmanaged<BadAppleWorkspaceWatcher>.fromOpaque(clientCallBackInfo).takeUnretainedValue()
            watcher.handleEvents(numEvents: numEvents, eventPaths: eventPaths, eventFlags: eventFlags, eventIds: eventIds)
        }

        guard let stream = FSEventStreamCreate(
            kCFAllocatorDefault,
            callback,
            &context,
            paths,
            FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            1.0,
            FSEventStreamCreateFlags(kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagUseCFTypes)
        ) else {
            watcherLog("failed to create FSEvent stream")
            return
        }

        FSEventStreamSetDispatchQueue(stream, DispatchQueue.main)
        if FSEventStreamStart(stream) {
            self.stream = stream
            running = true
            watcherLog("watching \"\(path)\"")
        } else {
            watcherLog("failed to start FSEvent stream")
            FSEventStreamRelease(stream)
        }
    }

    func stop() {
        DispatchQueue.main.async { [weak self] in
            self?._stop()
        }
    }

    private func _stop() {
        running = false
        lock.lock()
        pending.removeAll()
        flushWorkItem?.cancel()
        flushWorkItem = nil
        lock.unlock()

        if let stream {
            FSEventStreamStop(stream)
            FSEventStreamInvalidate(stream)
            FSEventStreamRelease(stream)
            self.stream = nil
        }
    }

    private func handleEvents(numEvents: Int, eventPaths: UnsafeMutableRawPointer, eventFlags: UnsafePointer<FSEventStreamEventFlags>, eventIds: UnsafePointer<FSEventStreamEventId>) {
        let cfArray = Unmanaged<CFArray>.fromOpaque(eventPaths).takeUnretainedValue() as! [String]
        lock.lock()
        defer { lock.unlock() }

        for (index, rawPath) in cfArray.enumerated() {
            let flags = Int(eventFlags[index])
            let isFile = (flags & kFSEventStreamEventFlagItemIsFile) != 0
            let isCreated = (flags & kFSEventStreamEventFlagItemCreated) != 0
            let isModified = (flags & kFSEventStreamEventFlagItemModified) != 0
            let isRenamed = (flags & kFSEventStreamEventFlagItemRenamed) != 0
            let isRemoved = (flags & kFSEventStreamEventFlagItemRemoved) != 0
            if isFile && (isCreated || isModified || isRenamed || isRemoved) {
                pending.insert(rawPath)
            }
        }

        if !pending.isEmpty {
            flushWorkItem?.cancel()
            let work = DispatchWorkItem { [weak self] in
                self?.flush()
            }
            flushWorkItem = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0, execute: work)
        }
    }

    private func flush() {
        lock.lock()
        let changed = pending
        pending.removeAll()
        flushWorkItem = nil
        lock.unlock()

        guard !changed.isEmpty else { return }
        watcherLog("re-indexing after changes in \(changed.count) file(s)")
        if let workspacePath = normalizePath(path) {
            indexCallback?(workspacePath)
        }
    }

    private func normalizePath(_ p: String) -> String? {
        var resolved = (p as NSString).standardizingPath
        if resolved.isEmpty { resolved = p }
        return resolved
    }
}
