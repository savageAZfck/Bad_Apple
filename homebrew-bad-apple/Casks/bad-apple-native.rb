cask "bad-apple-native" do
  version "0.9.9"
  # Update this sha256 for each release. package_minimal_release.sh
  # (BADAPPLE_NATIVE=1) prints the final hash.
  sha256 "5c7386e18457ae0d760eb1d1531f85fb5cd35242babfaca76ebb86842c6decca"

  url "https://github.com/savageAZfck/bad-apple-releases/releases/download/v#{version}/Bad_Apple-#{version}-native-unsigned.zip"
  name "Bad Apple (native organs)"
  desc "Sovereign, local AI operating-system layer — Coyote-free build, organs run natively in Rust"
  homepage "https://github.com/savageAZfck/bad-apple-releases"

  # The release zip contains both the .app bundle and the bad_apple platform.
  # Homebrew copies the app to /Applications; postflight copies the platform
  # to ~/.bad_apple/versions and runs the native platform installer.
  depends_on :macos
  depends_on arch: :arm64

  conflicts_with cask: "bad-apple"

  app "Bad_Apple-#{version}-native/Bad Apple.app"

  postflight_steps do
    # Keep a pristine, persistent copy of the platform per version so the
    # LaunchDaemons and menu bar app continue to work after Homebrew cleans
    # up the staged download. Create the versioned state root first: on a
    # cold install ~/.bad_apple does not exist yet and the copy fails.
    run "/bin/mkdir",
        args: ["-p", "#{Dir.home}/.bad_apple/versions/{{version}}"],
        must_succeed: true

    copy "Bad_Apple-{{version}}-native",
         ".bad_apple/versions/{{version}}/Bad_Apple-{{version}}-native",
         source_base: :staged_path,
         target_base: :home,
         recursive:   true,
         overwrite:   true

    run ".bad_apple/versions/{{version}}/Bad_Apple-{{version}}-native/install.sh",
        base:           :home,
        sudo:           true,
        print_stdout:   true,
        print_stderr:   true,
        must_succeed:   true,
        writable_paths: ["{{appdir}}/Bad Apple.app"],
        writable_base:  :home
  end

  uninstall_preflight_steps do
    run "/bin/launchctl",
        args: ["bootout", "system/com.badapple.supervisor"],
        sudo: true, must_succeed: false
    run "/bin/launchctl",
        args: ["bootout", "system/com.badapple.mlx"],
        sudo: true, must_succeed: false
    run "/bin/launchctl",
        args: ["bootout", "system/com.badapple.gatekeeper"],
        sudo: true, must_succeed: false
    run "/bin/rm",
        args: ["-f", "/usr/local/bin/badapple", "/usr/local/bin/badapple-fetch"],
        sudo: true, must_succeed: false
    # User LaunchAgents live in the console user's gui domain; boot them out
    # and remove the plists so no agent keeps pointing at a deleted payload.
    ["com.badapple.tts", "com.badapple.menubar", "com.badapple.dashboard",
     "com.badapple.checkpoint", "com.badapple.identity_agent"].each do |label|
      run "/bin/launchctl",
          args: ["bootout", "gui/#{Process.uid}/#{label}"],
          must_succeed: false
      run "/bin/rm",
          args: ["-f", "#{Dir.home}/Library/LaunchAgents/#{label}.plist"],
          must_succeed: false
    end
  end

  zap trash: [
    "/var/lib/bad_apple",
    "/var/log/bad_apple*.log",
    "~/.bad_apple",
  ]

  caveats <<~EOS
    Bad Apple is an unsigned, air-gapped app. Homebrew removes the Gatekeeper
    quarantine flag during install, but if you see a warning, run:
      xattr -dr com.apple.quarantine "/Applications/Bad Apple.app"

    The system daemons are installed during this cask. The main GPU brain
    (Qwen3-8B-4bit by default) is downloaded on first use only if you allow
    downloads in the menu bar or set BADAPPLE_ALLOW_DOWNLOADS=1. For an
    air-gap install, seed the model cache before the first query.

    This is the native-organs build: the organ layer runs in Rust through the
    same crates the Coyote organs were ported from — no .cy sources, no ANE
    bridge, identical semantics.
  EOS
end
