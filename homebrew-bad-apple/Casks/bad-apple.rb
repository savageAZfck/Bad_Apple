cask "bad-apple" do
  version "0.1.4"
  # Update this sha256 for each release. package_homebrew_cask.sh does it automatically.
  sha256 "6b965d969df49d16dfa8826c3802c53c4268d933cc4c06a267012ca0d8a4e31c"

  url "https://github.com/savageAZfck/bad-apple-releases/releases/download/v#{version}/Bad_Apple-#{version}-unsigned.zip"
  name "Bad Apple"
  desc "Sovereign, local AI operating-system layer"
  homepage "https://github.com/savageAZfck/bad-apple-releases"

  # The release zip contains both the .app bundle and the bad_apple platform.
  # Homebrew copies the app to /Applications; postflight copies the platform
  # to ~/.bad_apple/versions and runs the native platform installer.
  depends_on :macos
  depends_on arch: :arm64

  app "Bad_Apple-#{version}-unsigned/Bad Apple.app"

  postflight_steps do
    # Keep a pristine, persistent copy of the platform per version so the
    # LaunchDaemons and menu bar app continue to work after Homebrew cleans
    # up the staged download.
    copy "Bad_Apple-{{version}}-unsigned",
         ".bad_apple/versions/{{version}}/Bad_Apple-{{version}}-unsigned",
         source_base: :staged_path,
         target_base: :home,
         recursive:   true,
         overwrite:   true

    run ".bad_apple/versions/{{version}}/Bad_Apple-{{version}}-unsigned/install.sh",
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

    The system daemons are installed during this cask. The 7B model is
    downloaded on first use only if you allow downloads in the menu bar or set
    BADAPPLE_ALLOW_DOWNLOADS=1. For an air-gap install, seed the model cache
    before the first query.
  EOS
end
