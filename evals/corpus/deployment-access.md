# Synthetic fixture: MCP deployment access

This file is regression-test data, not a learned user memory.

Local MCP clients launch `sb-mcp` over stdio. A remote native client reaches the Streamable
HTTP endpoint at `http://YOUR_TAILSCALE_IP:8848/mcp` over Tailscale. HTTP clients send an
`Authorization: Bearer <token>` header. The endpoint should bind to the Tailscale address,
not `0.0.0.0`, unless LAN-wide access is intentional.
