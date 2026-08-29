# Two-Mac P2P test guide

This guide explains how to test Bad Apple's encrypted local mesh between two real Apple Silicon Macs. No cloud needed.

## What you need

- Two Apple Silicon Macs on the same Wi-Fi or Ethernet network.
- Bad Apple installed on both Macs.
- At least one small model cached on the first Mac (for example `mlx-community/Qwen2.5-0.5B-Instruct-4bit`).

## Step 1: Turn on P2P on both Macs

On each Mac, click the Bad Apple menu bar icon, choose **Mesh > P2P Sync** to turn it on.

Or type in Terminal:

```bash
badapple "turn on p2p"
```

## Step 2: Check that each Mac sees the other

On either Mac, open the Bad Apple menu and choose **Mesh > P2P Peers**, or type:

```bash
badapple p2p peers
```

You should see the other Mac in the list, like:

```json
{
  "peers_summary": "Discovered 1 peer:\n  Other-Mac@192.168.1.45:20000"
}
```

If no peer appears:
- Make sure both Macs are on the same local network.
- Make sure the firewall is not blocking local UDP/TCP. Bad Apple uses ports `19999` (UDP) and `20000` (TCP) by default.
- Restart the P2P daemon from the menu bar: **Troubleshooting > Restart Daemon**.

## Step 3: See what models the other Mac has

On the Mac that wants to receive a model, type:

```bash
badapple p2p models
```

You will see a list of models the other Mac has shared, for example:

```json
{
  "Other-Mac@192.168.1.45": [
    {
      "id": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
      "size_gb": 0.01,
      "provenance": "recorded"
    }
  ]
}
```

## Step 4: Pull the model manifest

To get the signed provenance record before copying any files:

```bash
badapple p2p pull "Other-Mac@192.168.1.45" "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

This downloads the manifest and checks the Secure Enclave signature. It does not copy the large weight files yet.

## Step 5: Transfer the model files

To copy the actual model weight files:

```bash
badapple p2p send "Other-Mac@192.168.1.45" "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

Run this on the Mac that already has the model. The receiving Mac must have P2P turned on and enough free disk space.

## Step 6: Verify the received model

On the receiving Mac, after the transfer finishes:

```bash
badapple model verify "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
```

Bad Apple will check every SHA-256 hash in the provenance manifest against the files it received. If the signature is valid and the hashes match, the model is safe to use.

## Troubleshooting

- **"No peers on the local network."**
  Both Macs must be on the same subnet. Some guest networks block device-to-device traffic. Use the main Wi-Fi network.

- **"Bad Apple could not find that peer."**
  The peer ID changed or the other Mac is no longer reachable. Run `badapple p2p peers` again to get the current ID.

- **Transfer is slow**
  Bad Apple uses TCP over your local network. Use Ethernet or a 5 GHz Wi-Fi network for the fastest transfer.

- **Not enough disk space**
  The receiving Mac needs free space equal to at least the model size. Use `badapple model info <id>` to see how big it is.
