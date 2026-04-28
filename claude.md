# Global AI CLI Agent Proxy Setup (Claude Code via NVIDIA NIM)

This machine is configured to run CLI agents (like Claude Code, Aider, Codex) completely **for free** by routing their traffic through an NVIDIA API Key using a **LiteLLM proxy**.

## 🚀 How it Works
1. When you type `claude` in your terminal, the CLI attempts to reach Anthropic's servers.
2. Global Windows Environment Variables (`ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`) intercept this request.
3. The request is routed to a local proxy running in the background (`http://localhost:4000`).
4. The proxy strips out Anthropic-specific formatting and forwards the prompt to **NVIDIA NIM**.
5. **Llama 3.1 70B** processes the prompt and returns the result back to your CLI!

## ⚙️ Configuration Location
The master configuration file that controls all of this is located at:
`C:\Users\umang\litellm_global_config.yaml`

If you ever need to change your NVIDIA API key, or if you want to switch from Llama 3 to a larger model (like `meta/llama-3.1-405b-instruct`), you simply edit that file.

## 🔄 How to Start the Proxy
If you reboot your computer or accidentally close the background proxy, your CLI agents will stop working. To fix it, simply open any terminal and run:

```powershell
$env:PYTHONIOENCODING="utf-8"
litellm --config "C:\Users\umang\litellm_global_config.yaml" --port 4000
```

*(Leave this window open in the background, or minimize it while you work in a different terminal)*

## ⚠️ Troubleshooting "Low Funds" Error
If Claude Code ever gives you a "Credit balance is too low" error, it means it is ignoring the proxy and trying to hit Anthropic directly using an old cached token.
**To fix this instantly:**
1. Run `claude logout`
2. Manually delete the hidden file `C:\Users\umang\.claude.json`
3. Restart your terminal window.
