const banner = document.querySelector("#banner");
const errorBox = document.querySelector("#error");
const thinking = document.querySelector("#thinking");
const answer = document.querySelector("#answer");
const stats = document.querySelector("#stats");
const kvState = document.querySelector("#kv-state");
const nodeState = document.querySelector("#node-state");
const macMeta = document.querySelector("#mac-meta");
const form = document.querySelector("#ask");
const send = document.querySelector("#send");

function paintPeer(node, ok, label) {
  node.textContent = ok ? `${label} reachable` : `${label} unreachable`;
  node.className = ok ? "range up" : "range down";
}

async function refresh() {
  const response = await fetch("/api/status");
  const status = await response.json();
  macMeta.textContent = `${status.engine} · ${status.model} · mode default ${status.mode}`;
  paintPeer(kvState, status.windows_kv === "up", "KV");
  paintPeer(nodeState, status.windows_node === "up", "node [14, 28)");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  banner.hidden = true;
  errorBox.hidden = true;
  send.disabled = true;
  thinking.textContent = "调度中…";
  answer.textContent = "";
  const mode = new FormData(form).get("mode");
  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: document.querySelector("#prompt").value,
        mode,
        max_new_tokens: Number(document.querySelector("#limit").value),
      }),
    });
    const body = await response.json();
    if (!response.ok) {
      errorBox.hidden = false;
      errorBox.textContent = body.error || "会话失败";
      thinking.textContent = "这次没有返回文本。dual 模式下 Windows 不可达会直接失败，不会改跑前半层。";
      answer.textContent = "";
      return;
    }
    thinking.textContent = body.thinking || "（没有 <think> 片段）";
    answer.textContent = body.answer || "（空回答）";
    stats.textContent = `mode ${body.mode_used} · degraded ${body.degraded || "—"} · prompt ${body.prompt_tokens} · completion ${body.completion_tokens}`;
    if (body.degraded === "mac_front_half") {
      banner.hidden = false;
      banner.textContent = body.warning;
    }
  } catch (err) {
    errorBox.hidden = false;
    errorBox.textContent = String(err);
  } finally {
    send.disabled = false;
    refresh().catch(() => {});
  }
});

refresh().catch(() => {
  kvState.textContent = "status failed";
  nodeState.textContent = "status failed";
});
