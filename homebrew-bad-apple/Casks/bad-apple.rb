cask "bad-apple" do
  version "0.1.0"
  # Update this sha256 for each release. package_homebrew_cask.sh does it automatically.
  sha256 "801d5fce09b2ad265af976bedaa836ebb0afe5c461fc375bfc88f1100c98dc2f"

  url "https://github.com/savage3/Bad_Apple/releases/download/v#{version}/Bad_Apple-#{version}-full-unsigned.zip"
  name "Bad Apple"
  desc "Air-gapped, on-device AI assistant"
  homepage "https://github.com/savage3/Bad_Apple"

  # The release zip contains both the .app bundle and the bad_apple platform.
  # Copy the app to /Applications; the postflight will stage the platform.
  depends_on :macos

  app "Bad_Apple-#{version}-full/Bad Apple.app"

  postflight do
    version = cask.version
    source = "#{staged_path}/Bad_Apple-#{version}-full/bad_apple"
    target = "#{Dir.home}/.bad_apple/versions/#{version}/bad_apple"

    # Keep a pristine copy of the platform directory per version.
    FileUtils.rm_r(target, force: true, verbose: false)
    FileUtils.mkdir_p File.dirname(target)
    FileUtils.cp_r source, target

    # Remove the Gatekeeper quarantine flag from the installed app.
    system_command "/usr/bin/xattr",
                   args:  ["-dr", "com.apple.quarantine", "#{appdir}/Bad Apple.app"],
                   print: false

    # Install the system LaunchDaemons. This will prompt for admin once.
    install_script = "#{target}/src/platform/apple_bridge/install_badapple_platform.sh"
    install_command = "\"#{install_script}\" --install --unsigned-install"
    system_command "/usr/bin/osascript",
                   args:  ["-e", "do shell script #{install_command} with administrator privileges"],
                   print: true
  end

  uninstall_preflight do
    # Unload the system LaunchDaemons before removing the app.
    system_command "/bin/launchctl",
                   args:         ["bootout", "system/com.badapple.supervisor"],
                   print:        false,
                   must_succeed: false
    system_command "/bin/launchctl",
                   args:         ["bootout", "system/com.badapple.mlx"],
                   print:        false,
                   must_succeed: false
    system_command "/bin/launchctl",
                   args:         ["bootout", "system/com.badapple.gatekeeper"],
                   print:        false,
                   must_succeed: false
  end

  zap trash: [
    "/var/lib/bad_apple",
    "/var/log/bad_apple*.log",
    "~/.bad_apple",
  ]

  caveats <<~EOS
    Bad Apple is an unsigned, air-gapped app. If you see a Gatekeeper warning,
    it should have been removed automatically. If not, run:
      xattr -dr com.apple.quarantine /Applications/Bad Apple.app

    The system daemons are installed during this cask. The 9B model is
    downloaded on first use.
  EOS
end
