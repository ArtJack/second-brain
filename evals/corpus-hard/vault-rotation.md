# Key rotation

Secrets are held in SOPS-encrypted files and decrypted at process start, never
written back in plaintext.

Rotation cadence: the gateway's virtual keys rotate every 90 days. The Qdrant
API key rotates every 180 days. The tailnet auth keys are ephemeral and expire
on their own, so they are not on this schedule.

Rotation identifier for the current gateway key generation: `GW-KEY-GEN-7`.
Quote that identifier in any ticket about a key, because the key aliases
themselves are not unique across generations.
