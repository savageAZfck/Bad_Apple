# The Bare-Metal AI Operating System Manifesto

## Preamble

We are building a new kind of operating system: a **cognitive layer that lives on the user's own hardware**, not in a rented data center. We call this a **bare-metal AGI operating system**.

The cloud AI industry wants you to believe that intelligence must be centralized: their GPUs, their terms, your data. We reject that premise. Intelligence can be local, private, fast, and sovereign.

This document is a declaration of the principles, promises, and interfaces that define a bare-metal AGI OS. It is not a product roadmap. It is a standard that any system may adopt, implement, or extend.

---

## The five principles

### 1. The user owns the machine, the machine owns the mind

The model, the memory, the tools, the policy, and the audit trail all run on the user's device. There is no multi-tenant server, no subscription, no API key, and no remote administrator. The user is the root of trust.

### 2. Air-gap is the default

A bare-metal AGI OS does not phone home. It does not send prompts, telemetry, or embeddings to a vendor. Network access is an explicit, auditable exception — never a default. If a model can be downloaded once and then cut off, it should be.

### 3. Every action is explainable and revocable

The assistant must be able to explain what it did, why it did it, and on what authority. Every tool call, every plan, every generated sentence can be traced back to a prompt, a policy, and a person. The user can approve, deny, or undo.

### 4. The cage is stronger than the brain

Capability without control is not intelligence — it is a liability. A bare-metal AGI OS embeds a policy layer that constrains the model before the model acts. The policy is human-readable, user-editable, and enforced by the OS, not by the assistant's good behavior.

### 5. Interoperability without centralization

A bare-metal AGI OS exposes open, local protocols so other apps and agents on the same device can extend its capabilities. It does not require a vendor's cloud to coordinate. It can talk to other bare-metal AGI OSes peer-to-peer, not platform-to-platform.

---

## What a bare-metal AGI OS must provide

| Layer | Responsibility | Non-negotiable properties |
|---|---|---|
| **Hardware root of trust** | Secure Enclave, biometric gate, device key, SLICKS identity | Secrets never leave the secure element. Identity is bound to the device. |
| **Local inference runtime** | MLX, ANE, NPU, or equivalent on-device model execution | No network during inference. Weights are cached and verified locally. |
| **Persona and memory system** | Identity, voice, style, long-term memory, skills | All stored locally. User can export, delete, or transfer. |
| **Tool and action system** | Shell, AppleScript, Shortcuts, file access, app automation | Governed by a declarative policy. Destructive actions require approval. |
| **Audit and provenance ledger** | Hash-chained, tamper-evident record of every action | Signed by the device. Verifiable offline. |
| **Streaming output firewall** | Real-time redaction of PII, secrets, and disallowed content | Runs between the model and the user. Cannot be bypassed by the model. |
| **Local agent protocol** | JSON-RPC / MCP / A2A over Unix socket with mutual auth | Only local clients. SLICKS-authenticated. No cloud broker. |
| **Certification suite** | Automated proof of air-gap, policy enforcement, and secret redaction | Run on every build. Failures block release. |

---

## What a bare-metal AGI OS must never do

1. **Never require a cloud account to function.**
2. **Never transmit user prompts, files, or embeddings to a third party.**
3. **Never hide its network behavior, its policy, or its audit log.**
4. **Never allow the model to override the cage.**
5. **Never make the user dependent on a vendor's continued goodwill.**

---

## The promise to the user

When you run a bare-metal AGI OS:

- Your data stays on your hardware.
- Your assistant is yours.
- Your secrets are not training data.
- Your actions are recorded, not reported.
- You can see the rules, change the rules, and prove the rules were followed.

---

## The promise to builders

A bare-metal AGI OS is not a walled garden. It is a reference architecture. If you implement the interfaces, follow the manifesto, and pass the certification suite, your system is a bare-metal AGI OS. You can extend it, sell it, fork it, or integrate it — as long as you keep the user sovereign.

---

## Against the cloud AI model

The cloud AI model is:

- **Rented.** You do not own the weights, the memory, or the infrastructure.
- **Surveilled.** Your data is a product, a training target, and a liability.
- **Fragile.** A subscription change, a terms-of-service update, or an API deprecation can break your workflow overnight.
- **Centralized.** A single breach, a single law, or a single outage affects millions.

A bare-metal AGI OS is the opposite: **owned, private, durable, and distributed.**

---

## Call to builders

The cloud will not save us from the problems it created. If we want AI that respects the user, we must build it on the user's own metal.

This manifesto is a starting point. The code is the proof. The standard is the commitment.

**Bad Apple** is the first implementation. It will not be the last.
