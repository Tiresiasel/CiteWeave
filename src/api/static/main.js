const API_BASE = '/api/v1';

function $(q) { return document.querySelector(q); }
function ce(tag, className) { const el = document.createElement(tag); if (className) el.className = className; return el; }

const state = {
  chats: JSON.parse(localStorage.getItem('cw_chats') || '[]'),
  activeChatId: localStorage.getItem('cw_active_chat') || null,
  settings: JSON.parse(localStorage.getItem('cw_settings') || '{}'),
  docs: [],
  collections: JSON.parse(localStorage.getItem('cw_collections') || '[]'),
  selectedCollection: localStorage.getItem('cw_selected_collection') || null,
  route: location.hash || '#/chat',
};

function saveState() {
  localStorage.setItem('cw_chats', JSON.stringify(state.chats));
  localStorage.setItem('cw_active_chat', state.activeChatId || '');
  localStorage.setItem('cw_settings', JSON.stringify(state.settings));
  localStorage.setItem('cw_collections', JSON.stringify(state.collections || []));
  localStorage.setItem('cw_selected_collection', state.selectedCollection || '');
}

function newChat(title = 'New chat') {
  const id = 'chat_' + Date.now();
  const chat = { id, title, history: [], collected_data: null, created_at: Date.now() };
  state.chats.unshift(chat);
  state.activeChatId = id;
  saveState();
  renderRoute();
}

function getActiveChat() {
  return state.chats.find(c => c.id === state.activeChatId) || null;
}

function getUserAvatarHtml() {
  const url = state.settings && state.settings.user_avatar_url;
  if (url) return `<div class="avatar"><img src="${url}" alt="User"/></div>`;
  // Fallback to the top-right avatar initial
  const top = document.getElementById('btn_account');
  const ch = (top && (top.textContent || '').trim()[0]) || (state.settings && state.settings.display_name ? state.settings.display_name.trim()[0] : 'A');
  return `<div class="avatar"><div style=\"width:100%;height:100%;display:flex;align-items:center;justify-content:center;background:#1f2937;color:#e5e7eb;font-weight:700\">${(ch || 'A').toUpperCase()}</div></div>`;
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

let __libRenderToken = 0; // guards against overlapping renders
const __fetchControllers = { collections: null, settings: null, jobs: null, documents: null };
function __beginFetch(key){
  try { __fetchControllers[key]?.abort?.(); } catch(_){}
  const c = new AbortController();
  __fetchControllers[key] = c;
  return c.signal;
}

async function listDocs() {
  try {
    const col = state.selectedCollection || 'Default';
    const data = await apiFetch('/documents?collection=' + encodeURIComponent(col));
    state.docs = data.documents || [];
    if (state.route === '#/library') renderLibrary();
  } catch (e) {
    console.error('List docs failed', e);
  }
}

async function listCollections(signal) {
  try {
    const data = await apiFetch('/collections', { signal });
    const cols = (data.collections || []).map(c => c.name);
    // ensure Default exists visibly
    if (!cols.includes('Default')) cols.unshift('Default');
    state.collections = cols;
    if (!state.selectedCollection) {
      state.selectedCollection = cols.includes('Default') ? 'Default' : (cols[0] || 'Default');
    }
    saveState();
  } catch (e) {
    // fallback to Default locally
    if (!state.collections.length) state.collections = ['Default'];
    if (!state.selectedCollection) state.selectedCollection = 'Default';
    saveState();
  }
}

async function sendMessage(text) {
  const chat = getActiveChat();
  if (!chat) return;
  chat.history.push({ user: text, ai: '' });
  renderChat(true);
  try {
    const lastTurn = chat.history[chat.history.length - 1];
    const payload = {
      user_input: text,
      history: chat.history.slice(0, -1),
      collected_data: chat.collected_data,
    };
    const resp = await apiFetch('/chat', { method: 'POST', body: JSON.stringify(payload) });
    const r = resp.response || {};
    lastTurn.ai = r.text || '';
    chat.collected_data = r.collected_data || null;
    chat.bibliography = r.bibliography || [];
    saveState();
    renderRoute();
    renderRefsPanel();
  } catch (e) {
    console.error('Chat failed', e);
  }
}

async function sendMenuChoice(choice) {
  const chat = getActiveChat();
  if (!chat) return;
  const lastAI = chat.history[chat.history.length - 1]?.ai || '';
  chat.history.push({ user: `[choice ${choice}]`, ai: '' });
  renderChat();
  try {
    const payload = {
      user_input: lastAI || 'continue',
      history: chat.history.slice(0, -1),
      menu_choice: String(choice),
      collected_data: chat.collected_data,
    };
    const resp = await apiFetch('/chat', { method: 'POST', body: JSON.stringify(payload) });
    const r = resp.response || {};
    chat.history[chat.history.length - 1].ai = r.text || '';
    chat.collected_data = r.collected_data || null;
    chat.menu = r.needs_user_choice ? r.menu || [] : null;
    saveState();
    renderRoute();
    if (chat.menu && chat.menu.length) renderMenu(chat.menu);
  } catch (e) {
    console.error('Menu choice failed', e);
  }
}
function renderChatPage() {
  const view = $('#view');
  view.innerHTML = '';
  const grid = ce('div', 'container');
  const left = ce('div', 'card left-pane');
  left.innerHTML = `
    <div class="header">Chats</div>
    <div class="body">
      <button id="btn_new_chat" class="primary" style="width:100%; margin-bottom:8px">New chat</button>
      <div class="section-title" style="margin-top:4px">Recent</div>
      <div id="chat_list" class="list"></div>
    </div>`;
  grid.appendChild(left);
  const center = ce('div', 'card chat-pane center-pane');
  center.innerHTML = `
    <div class="messages" id="chat"></div>
    <div class="composer" style="position:sticky; bottom:0;">
      <div class="input-wrap">
        <textarea id="input" placeholder="Ask anything..."></textarea>
        <button id="send" class="send-btn" title="Send" aria-label="Send">
          <svg class="send-triangle" width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor">
            <path d="M5 3l15 9-15 9 4-9-4-9z"/>
          </svg>
        </button>
      </div>
    </div>
    <button id="btn_scroll_bottom" class="scroll-bottom hidden" title="Scroll to bottom" aria-label="Scroll to bottom">↓</button>
  `;
  grid.appendChild(center);
  // Right: References (titles only)
  const right = ce('div', 'card right-pane');
  right.innerHTML = `<div class="header">References</div><div class="body"><div id="refs" class="list"></div></div>`;
  grid.appendChild(right);
  $('#view').appendChild(grid);

  $('#btn_new_chat').onclick = () => newChat();
  $('#send').onclick = () => { const v = $('#input').value.trim(); if (v) sendMessage(v); $('#input').value=''; };
  $('#input').addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); $('#send').click(); } else if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){ $('#send').click(); }});

  renderSidebar();
  renderChat();
  renderRefsPanel();

  // scroll-to-bottom visibility & action
  const chatBox = document.getElementById('chat');
  const sb = document.getElementById('btn_scroll_bottom');
  const toggleSB = () => {
    if (!chatBox) return;
    const nearBottom = chatBox.scrollHeight - chatBox.scrollTop - chatBox.clientHeight < 80;
    sb.classList.toggle('hidden', nearBottom);
  };
  chatBox.addEventListener('scroll', toggleSB);
  sb.onclick = ()=> { chatBox.scrollTop = chatBox.scrollHeight; };
  setTimeout(()=>{ chatBox.scrollTop = chatBox.scrollHeight; toggleSB(); }, 0);
}

function renderLibrary() {
  const token = ++__libRenderToken;
  console.log(`[DEBUG] renderLibrary started with token: ${token}`);
  const view = $('#view');
  view.innerHTML = '';
  const grid = ce('div', 'container-library');
  // Left: Collections
  const left = ce('div', 'card');
  left.innerHTML = `
    <div class="header">Collections</div>
    <div class="body">
      <div class="row" style="margin-bottom:10px">
        <button id="add_collection" class="primary" style="width:100%">+ New collection</button>
      </div>
      <div id="collections_list" class="list"></div>
      <div class="row" style="margin-top:10px; display:none">
        <button id="rename_collection" class="btn" title="Rename collection">Rename</button>
      </div>
    </div>`;
  grid.appendChild(left);

  // Right: Upload & Settings
  const right = ce('div', 'card');
  right.innerHTML = `
    <div class="header">Upload</div>
    <div class="body">
      <div class="row" style="justify-content: space-between; margin-bottom:8px">
        <div>
          <span class="pill active" id="tab_files">Files</span>
          <span class="pill" id="tab_folders" style="margin-left:8px">Folders</span>
        </div>
        <!-- Embedding selection moved to Settings; removed from Library top bar -->
      </div>
      <div id="upload_panel">
        <div id="uploader" class="uploader">Drag & drop files here (PDF)</div>
        <div id="upload_list" class="list" style="margin-top:12px"></div>
        <div class="section-title" style="margin-top:14px">Documents</div>
        <div id="docs_list" class="list" style="margin-top:6px"></div>
      </div>
      <div id="folder_panel" style="display:none">
        <div id="folder_inputs" class="list" style="margin-top:10px; gap:8px"></div>
        <div id="watch_list" class="list" style="margin-top:12px"></div>
      </div>
    </div>`;
  grid.appendChild(right);
  view.appendChild(grid);

  // Always reset runtime caches when entering Library to avoid stale UI
  state.docs = [];
  const uploadList = document.getElementById('upload_list');
  const docsList = document.getElementById('docs_list');
  if (uploadList) {
    console.log(`[DEBUG] Clearing upload_list: had ${uploadList.children.length} children`);
    uploadList.innerHTML = '';
  }
  if (docsList) {
    console.log(`[DEBUG] Clearing docs_list: had ${docsList.children.length} children`);
    docsList.innerHTML = '';
  }

  // Event delegation for document rows (ensures clicks work after re-renders)
  const docsEl = document.getElementById('docs_list');
  if (docsEl && !docsEl.dataset.wired) {
    docsEl.dataset.wired = '1';
    docsEl.addEventListener('click', async (ev) => {
      const chip = ev.target.closest('.icon-chip');
      if (!chip) return;
      const row = chip.closest('.row');
      const pid = row && row.dataset && row.dataset.paperId;
      if (!pid) return; // ignore upload job rows (no paper id yet)
      const statusChip = row.querySelector('.icon-chip.status');
      const base = state.settings.api_base || '';
      if (chip.classList.contains('warn')) {
        // Delete
        try {
          const r = await fetch((base||'') + API_BASE + '/documents/' + pid + '/delete', { method:'POST' });
          const jr = await r.json().catch(()=>({}));
          if (r.ok && jr && jr.success) {
            row.remove();
          } else {
            alert('Delete failed');
          }
        } catch(e) { alert('Delete failed'); }
        return;
      }
      if (chip.getAttribute('aria-label') === 'Reprocess') {
        try {
          statusChip.classList.remove('done','error');
          statusChip.title = 'Pending';
          const ic = statusChip.querySelector('svg.icon');
          if (ic) ic.outerHTML = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='6' stroke='currentColor' stroke-width='1.6'/></svg>";
          const ring = statusChip.querySelector('svg.ring'); if (ring) ring.style.display = '';
          const r = await fetch((base||'') + API_BASE + '/docs/reprocess', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ paper_id: pid }) });
          const resp = await r.json();
          if (!r.ok || !resp.success || !resp.job_id) { alert('Reprocess failed'); return; }
          const jobId = resp.job_id;
          const poll = async ()=>{
            try {
              const jr = await fetch((base||'') + API_BASE + '/jobs/' + jobId);
              const jd = await jr.json();
              if (!jr.ok || !jd.success) return setTimeout(poll, 1200);
              const pr = Math.max(0, Math.min(100, jd.job.progress || 0));
              const arc2 = row.querySelector('.icon-chip.status .arc');
              if (arc2) { const end=(pr/100)*2*Math.PI; const r=16, cx=18, cy=18; const x=cx + r*Math.sin(end), y=cy - r*Math.cos(end); const large=end>Math.PI?1:0; arc2.setAttribute('d', `M18 2 a16 16 0 ${large} 1 ${x-18} ${y-2}`); arc2.setAttribute('stroke', pr>=100?'#22c55e':'#6ee7b7'); }
              if (!jd.job.done) return setTimeout(poll, 1000);
              if (jd.job.success) { statusChip.classList.add('done'); statusChip.title='Done'; const ic2=statusChip.querySelector('svg.icon'); if (ic2) ic2.outerHTML = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M5 13l4 4L19 7' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/></svg>"; }
              else { statusChip.classList.add('error'); statusChip.title='Failed'; const ic2=statusChip.querySelector('svg.icon'); if (ic2) ic2.outerHTML = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M7 7l10 10M17 7l-10 10' stroke='currentColor' stroke-width='1.8' stroke-linecap='round'/></svg>"; }
            } catch(e) { setTimeout(poll, 1200); }
          };
          poll();
        } catch(e) { alert('Reprocess failed'); }
      }
    });
  }

  // Fallback: ensure UI shows a Default collection immediately
  if (!state.collections || !state.collections.length) {
    state.collections = ['Default'];
    if (!state.selectedCollection) state.selectedCollection = 'Default';
  }
  renderCollections(state.collections);

  // Add collection button
  const addBtn = left.querySelector('#add_collection');
  if (addBtn) {
    addBtn.onclick = ()=> beginInlineNewCollection();
  }

  // Tab switching
  const tabFiles = document.getElementById('tab_files');
  const tabFolders = document.getElementById('tab_folders');
  const uploadPanel = document.getElementById('upload_panel');
  const folderPanel = document.getElementById('folder_panel');
  tabFiles.onclick = ()=>{ tabFiles.classList.add('active'); tabFolders.classList.remove('active'); uploadPanel.style.display='block'; folderPanel.style.display='none'; };
  tabFolders.onclick = ()=>{ tabFolders.classList.add('active'); tabFiles.classList.remove('active'); uploadPanel.style.display='none'; folderPanel.style.display='block'; };

  // Drag-and-drop
    const dz = document.getElementById('uploader');
  if (dz) {
    dz.addEventListener('dragover', (e)=>{ e.preventDefault(); e.stopPropagation(); dz.classList.add('dragover'); });
    dz.addEventListener('dragleave', (e)=>{ e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover'); });
    dz.addEventListener('drop', async (e)=>{
      e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover');
      const files = Array.from(e.dataTransfer.files || []).filter(f=>/\.pdf$/i.test(f.name));
      if (!files.length) return;
      // show pending items in list UI immediately and start async processing
      const list = document.getElementById('upload_list');
      const base = state.settings.api_base || '';
      for (const f of files) {
        const row = ce('div', 'row');
        row.style.justifyContent = 'space-between';
        row.style.alignItems = 'center';
        row.style.padding = '8px 10px';
        row.style.border = '1px solid var(--muted)';
        row.style.borderRadius = '12px';
        row.style.marginBottom = '6px';
        row.innerHTML = `<div style="color:var(--text)">${f.name}</div>
          <div class="row" style="gap:8px; align-items:center">
            <div class="icon-chip" title="Reprocess" role="button" aria-label="Reprocess">
              <svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M3 12a9 9 0 1 1 2.64 6.36' stroke='currentColor' stroke-width='1.6'/><path d='M3 16v-4h4' stroke='currentColor' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/></svg>
            </div>
            <div class="icon-chip status" id="status" title="Pending">
              <svg class="ring" viewBox="0 0 36 36"><path class="track" d="M18 2a16 16 0 1 1 0 32 16 16 0 0 1 0-32"/><path class="arc" d="M18 2 a16 16 0 0 1 0 0"/></svg>
              <svg class="icon" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='6' stroke='currentColor' stroke-width='1.6'/></svg>
            </div>
            <div class="icon-chip warn" title="Delete" role="button" aria-label="Delete">
              <svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M5 7h14M9 7V5h6v2m-8 0l1 12h8l1-12' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg>
            </div>
          </div>`;
        const reprocessChip = row.querySelector('.icon-chip');
        const statusChip = row.querySelector('#status');
        const deleteChip = row.querySelector('.icon-chip.warn');
        list.appendChild(row);

        // start async upload job
        let paperId = null;
        try {
          const fd = new FormData(); fd.append('file', f);
          // attach current collection so backend writes it
          const col = state.selectedCollection || 'Default';
          fd.append('collection', col);
          const res = await fetch((base||'') + API_BASE + '/upload_async', { method:'POST', body: fd });
          const data = await res.json();
          if (res.ok && data.success) {
            const jobId = data.job_id;
            const poll = async ()=>{
              const jr = await fetch((base||'') + API_BASE + '/jobs/' + jobId);
              const jd = await jr.json();
              if (!jr.ok || !jd.success) return setTimeout(poll, 1200);
              // update circular arc
              const pr = Math.max(0, Math.min(100, jd.job.progress || 0));
              const arc = row.querySelector('.icon-chip.status .arc');
              if (arc) {
                const end = (pr/100)*2*Math.PI; const r=16, cx=18, cy=18; const x=cx + r*Math.sin(end), y=cy - r*Math.cos(end); const large = end > Math.PI ? 1 : 0; arc.setAttribute('d', `M18 2 a16 16 0 ${large} 1 ${x-18} ${y-2}`); arc.setAttribute('stroke', pr>=100 ? '#22c55e' : '#6ee7b7');
              }
              if (!jd.job.done) return setTimeout(poll, 1000);
              // Once job finishes (success or failure), remove this transient row.
              if (jd.job.success) {
                paperId = jd.job.result?.paper_id || jd.job.summary?.paper_id || null;
                await listDocs();
              }
              try { row.remove(); } catch(_) {}
            };
            poll();
          }
        } catch (err) { console.error('async upload failed', err); }

        // no manual toggle for status chip (avoid confusing double icons)

        // reprocess
        reprocessChip.onclick = async ()=>{
          if (!paperId) return;
          statusChip.classList.remove('done','error');
          statusChip.title = 'Pending';
          statusChip.querySelector('svg.icon').outerHTML = "<svg class='icon' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='6' stroke='currentColor' stroke-width='1.6'/></svg>";
          try { await apiFetch('/docs/reprocess', { method:'POST', body: JSON.stringify({ paper_id: paperId }) }); } catch(e){}
        };

        // delete
        deleteChip.onclick = async ()=>{
          if (!paperId) return;
          try { await fetch((base||'') + API_BASE + '/docs/' + paperId, { method:'DELETE' }); row.remove(); await listDocs(); } catch(e){}
        };
      }
    });
  }

  // Buttons
  // No explicit button upload; drag-and-drop only for files

  // Create the first folder input row
  addFolderInputRow();

  // legacy handler removed; use inline new collection editor instead
  const legacyAdd = document.getElementById('add_collection');
  if (legacyAdd) legacyAdd.onclick = ()=> beginInlineNewCollection();

  // Embedding provider select -> persist setting
  // Embedding provider selection is configured in Settings page; no selector in Library

  // Prefill collections (from backend) and embedding setting
  (async ()=>{
    try {
      const sigCols = __beginFetch('collections');
      await listCollections(sigCols);
      if (token !== __libRenderToken) return; // outdated render
      renderCollections(state.collections);
      if (!state.collections || !state.collections.length) {
        state.collections = ['Default'];
        if (!state.selectedCollection) state.selectedCollection = 'Default';
        if (token !== __libRenderToken) return;
        renderCollections(state.collections);
      }
      const sigSet = __beginFetch('settings');
      const s = await apiFetch('/settings', { signal: sigSet });
      const v = s.settings || {};
      if (token !== __libRenderToken) return;
      await renderWatchList();
    } catch(e){
      if (token !== __libRenderToken) return;
      renderCollections(state.collections.length ? state.collections : ['Default']);
    }
    // Always rebuild lists fresh
    const ul = document.getElementById('upload_list');
    if (ul) ul.innerHTML = '';
    try {
      if (token !== __libRenderToken) return;
      const sigJobs = __beginFetch('jobs');
      await restoreUploadJobs(sigJobs, token);
    } catch(e) { console.error('restore jobs failed', e); }
    // Hard-clear any previously appended rows before rendering docs (safety)
    if (docsList) {
      console.log(`[DEBUG] Before docs render: clearing docs_list with ${docsList.children.length} children`);
      docsList.innerHTML = '';
    }
    try {
      const col = state.selectedCollection || 'Default';
      const sigDocs = __beginFetch('documents');
      const resp = await apiFetch('/documents?collection=' + encodeURIComponent(col), { signal: sigDocs });
      state.docs = resp.documents || [];
      console.log(`[DEBUG] Fetched ${state.docs.length} documents from backend for collection: ${col}`);
      if (token !== __libRenderToken) return;
      const seenDocs = new Set();
      const frag = document.createDocumentFragment();
      let addedDocCount = 0;
      (state.docs || []).forEach(d => {
        if (token !== __libRenderToken) return; // guard outdated loop items
        if (!d || !d.paper_id) return;
        if (seenDocs.has(d.paper_id)) return;
        seenDocs.add(d.paper_id);
        const row = ce('div', 'row');
        row.style.justifyContent = 'space-between';
        row.style.alignItems = 'center';
        row.style.padding = '8px 10px';
        row.style.border = '1px solid var(--muted)';
        row.style.borderRadius = '12px';
        row.style.marginBottom = '6px';
        const displayName = (d.title && d.title.trim()) || (d.new_filename ? d.new_filename.replace(/\.pdf$/i,'') : '') || d.original_filename || d.paper_id;
        // Map backend status to icon and class
        const s = String(d.status || '').toLowerCase();
        const isDone = s === 'done' || s === 'success' || s === 'completed';
        const isError = s === 'error' || s === 'failed' || s === 'failure';
        const statusClass = `icon-chip status${isDone ? ' done' : ''}${isError ? ' error' : ''}`;
        const iconDone = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M5 13l4 4L19 7' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'/></svg>";
        const iconFail = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M7 7l10 10M17 7l-10 10' stroke='currentColor' stroke-width='2' stroke-linecap='round'/></svg>";
        const iconHollow = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='6' stroke='currentColor' stroke-width='1.6'/></svg>";
        const centerIcon = isDone ? iconDone : isError ? iconFail : iconHollow;
        const titleText = isDone ? 'Done' : isError ? 'Failed' : 'Pending';
        row.innerHTML = `<div style=\"color:var(--text)\">${displayName}</div>
          <div class="row" style="gap:8px; align-items:center">
            <div class="icon-chip" title="Reprocess" role="button" aria-label="Reprocess">
              <svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M3 12a9 9 0 1 1 2.64 6.36' stroke='currentColor' stroke-width='1.8'/><path d='M3 16v-4h4' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/></svg>
            </div>
            <div class="${statusClass}" title="${titleText}">
              <svg class="ring" viewBox="0 0 36 36"><path class="track" d="M18 2a16 16 0 1 1 0 32 16 16 0 0 1 0-32"/><path class="arc" d="M18 2 a16 16 0 0 1 0 0"/></svg>
              ${centerIcon}
            </div>
            <div class="icon-chip warn" title="Delete" role="button" aria-label="Delete">
              <svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M5 7h14M9 7V5h6v2m-8 0l1 12h8l1-12' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg>
            </div>`;
        frag.appendChild(row);
        addedDocCount++;
        console.log(`[DEBUG] Added doc row ${addedDocCount}: ${displayName} (paper_id: ${d.paper_id})`);
        // handlers
        const reprocess = row.querySelector('.icon-chip[aria-label="Reprocess"]');
        const statusChip = row.querySelector('.icon-chip.status');
        reprocess.onclick = async ()=>{
          try {
            // visual reset to pending
            statusChip.classList.remove('done','error');
            statusChip.title = 'Pending';
            const ic = statusChip.querySelector('svg.icon');
            if (ic) ic.outerHTML = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='6' stroke='currentColor' stroke-width='1.6'/></svg>";
            // show ring again for progress
            const ring = statusChip.querySelector('svg.ring');
            if (ring) ring.style.display = '';
            const base = state.settings.api_base || '';
            const r = await fetch((base||'') + API_BASE + '/docs/reprocess', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ paper_id: d.paper_id }) });
            const resp = await r.json();
            if (!r.ok || !resp.success || !resp.job_id) return;
            const jobId = resp.job_id;
            const poll = async ()=>{
              try {
                const jr = await fetch((base||'') + API_BASE + '/jobs/' + jobId);
                const jd = await jr.json();
                if (!jr.ok || !jd.success) return setTimeout(poll, 1200);
                const pr = Math.max(0, Math.min(100, jd.job.progress || 0));
                const arc2 = row.querySelector('.icon-chip.status .arc');
                if (arc2) { const end=(pr/100)*2*Math.PI; const r=16, cx=18, cy=18; const x=cx + r*Math.sin(end), y=cy - r*Math.cos(end); const large=end>Math.PI?1:0; arc2.setAttribute('d', `M18 2 a16 16 0 ${large} 1 ${x-18} ${y-2}`); arc2.setAttribute('stroke', pr>=100?'#22c55e':'#6ee7b7'); }
                if (!jd.job.done) return setTimeout(poll, 1000);
                if (jd.job.success) { statusChip.classList.add('done'); statusChip.title='Done'; const ic2=statusChip.querySelector('svg.icon'); if (ic2) ic2.outerHTML = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M5 13l4 4L19 7' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'/></svg>"; }
                else { statusChip.classList.add('error'); statusChip.title='Failed'; const ic2=statusChip.querySelector('svg.icon'); if (ic2) ic2.outerHTML = "<svg class=\"icon\" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M7 7l10 10M17 7l-10 10' stroke='currentColor' stroke-width='1.8' stroke-linecap='round'/></svg>"; }
              } catch(e) { setTimeout(poll, 1200); }
            };
            poll();
          } catch(e) { /* ignore */ }
        };
        // delete handler
        const del = row.querySelector('.icon-chip.warn');
        del.onclick = async ()=>{
          try {
            const base = state.settings.api_base || '';
            const r = await fetch((base||'') + API_BASE + '/documents/' + d.paper_id + '/delete', { method:'POST' });
            const jr = await r.json().catch(()=>({}));
            if (r.ok && jr && jr.success) {
              row.remove();
              await listDocs();
            }
          } catch(e){}
        };
      });
      console.log(`[DEBUG] Adding ${addedDocCount} document rows to DOM`);
      const listNow = document.getElementById('docs_list');
      if (listNow) {
        console.log(`[DEBUG] Before appending fragment: docs_list has ${listNow.children.length} children`);
        listNow.appendChild(frag);
        console.log(`[DEBUG] After appending fragment: docs_list has ${listNow.children.length} children`);
      }
    } catch(e) { console.error('load docs to list failed', e); }
  })();
}

function addFolderInputRow(prefillPath=""){
  const wrap = document.getElementById('folder_inputs');
  if (!wrap) return;
  const row = document.createElement('div');
  row.className = 'row';
  row.style.gap = '8px';
  row.style.alignItems = 'center';
  const input = document.createElement('input');
  input.className = 'select';
  input.placeholder = '/path/to/folder';
  input.style.flex = '1.2';
  input.value = prefillPath;
  const btn = document.createElement('button');
  btn.className = 'primary';
  btn.textContent = 'Add';
  btn.style.flex = '0 0 90px';
  btn.style.padding = '8px 10px';
  btn.onclick = async ()=>{
    const path = (input.value||'').trim();
    if (!path) { input.focus(); return; }
    try {
      const col = state.selectedCollection || 'Default';
      const s = await apiFetch('/settings');
      const v = s.settings || {};
      const map = Array.isArray(v.watch_map) ? v.watch_map.slice() : [];
      if (!map.find(e=>e.path===path)) map.push({ path, collection: col });
      await apiFetch('/settings', { method:'POST', body: JSON.stringify({ watch_map: map, watch_enabled: true }) });
      // button state: Saved, then add a new empty row below
      btn.textContent = 'Saved';
      btn.disabled = true;
      // trigger immediate scan
      try { await apiFetch('/watch/scan-now', { method:'POST' }); } catch(_){}
      addFolderInputRow("");
      await renderWatchList();
    } catch(e) { console.error(e); }
  };
  row.appendChild(input); row.appendChild(btn);
  wrap.appendChild(row);
}

function renderCollections(cols) {
  const list = document.getElementById('collections_list');
  if (!list) return;
  list.innerHTML = '';
  const current = state.selectedCollection || 'Default';
  (cols || ['Default']).forEach(name => {
    const btn = ce('button', 'btn');
    btn.textContent = name;
    if (name === current) btn.classList.add('btn-active');
    btn.onclick = async ()=>{
      state.selectedCollection = name;
      saveState();
      renderCollections(cols);
      await listDocs();
    };
    // Inline rename on double click (仍可用)
    btn.ondblclick = ()=> beginInlineRename(btn, name);

    const wrap = ce('div');
    wrap.style.position = 'relative';
    wrap.appendChild(btn);

    if (name !== 'Default') {
      const dots = ce('div');
      dots.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/></svg>';
      dots.style.position = 'absolute';
      // Slightly more left; visually centered vertically
      dots.style.right = '10px';
      dots.style.top = '53%';
      dots.style.transform = 'translateY(-50%)';
      dots.style.display = 'none';
      dots.style.cursor = 'pointer';
      dots.style.width = '20px';
      dots.style.height = '20px';
      dots.style.display = 'none';
      dots.style.alignItems = 'center';
      dots.style.justifyContent = 'center';
      wrap.onmouseenter = ()=>{ dots.style.display = ''; };
      wrap.onmouseleave = ()=>{ dots.style.display = 'none'; };
      dots.onclick = (e)=>{ e.stopPropagation(); showCollectionMenu(dots, name, btn, cols); };
      wrap.appendChild(dots);
    }

    list.appendChild(wrap);
  });
}

function beginInlineNewCollection() {
  const list = document.getElementById('collections_list');
  if (!list) return;
  // avoid duplicating an input if already present
  if (list.querySelector('input.__col_edit')) return;
  const wrap = ce('div');
  const input = ce('input', 'select __col_edit');
  input.placeholder = 'New collection name';
  input.style.width = '100%';
  wrap.appendChild(input);
  list.appendChild(wrap);
  input.focus();
  const commit = async ()=>{
    const val = (input.value || '').trim();
    wrap.remove();
    if (!val) return;
    try {
      await apiFetch('/collections', { method:'POST', body: JSON.stringify({ name: val }) });
      await listCollections(); // backend returns list ordered by created_at ASC
      state.selectedCollection = val;
      saveState();
      renderCollections(state.collections);
      await listDocs();
    } catch(e) { alert('Create failed'); }
  };
  input.addEventListener('keydown', (e)=>{ if (e.key==='Enter') commit(); if (e.key==='Escape'){ wrap.remove(); }});
  input.addEventListener('blur', commit);
}

function beginInlineRename(btn, oldName) {
  const list = document.getElementById('collections_list');
  if (!list) return;
  const idxNode = btn;
  const input = ce('input', 'select __col_edit');
  input.value = oldName;
  input.style.width = btn.offsetWidth ? (btn.offsetWidth + 'px') : '100%';
  // When items are wrapped, replace the wrapper instead of the button
  const targetNode = idxNode.parentElement && idxNode.parentElement.parentElement === list ? idxNode.parentElement : idxNode;
  list.replaceChild(input, targetNode);
  input.focus();
  const cancel = ()=>{ try { list.replaceChild(targetNode, input); } catch(_){} };
  const commit = async ()=>{
    const val = (input.value || '').trim();
    if (!val || val === oldName) { cancel(); return; }
    try {
      await apiFetch('/collections/rename', { method:'POST', body: JSON.stringify({ old_name: oldName, new_name: val }) });
      await listCollections();
      if (state.selectedCollection === oldName) state.selectedCollection = val;
      saveState();
      renderCollections(state.collections);
      await listDocs();
    } catch(e) { alert('Rename failed'); cancel(); }
  };
  input.addEventListener('keydown', (e)=>{ if (e.key==='Enter') commit(); if (e.key==='Escape') cancel(); });
  input.addEventListener('blur', commit);
}

let __openMenuEl = null;
function closeOpenMenu(){ if (__openMenuEl){ __openMenuEl.remove(); __openMenuEl=null; } }

function showCollectionMenu(anchor, name, btnRef, cols){
  closeOpenMenu();
  const menu = ce('div');
  menu.className = 'context-menu';
  menu.style.position = 'fixed';
  const rect = anchor.getBoundingClientRect();
  menu.style.left = (rect.left - 6) + 'px';
  menu.style.top = (rect.bottom + 6) + 'px';
  menu.style.zIndex = '2000';
  menu.innerHTML = `
    <div class="list">
      <div class="item" id="mi_rename">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z" stroke="currentColor" stroke-width="1.6"/></svg>
        <span>Rename</span>
      </div>
      <div class="item warn" id="mi_delete">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none"><path d="M5 7h14M9 7V5h6v2m-8 0l1 12h8l1-12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
        <span>Delete</span>
      </div>
    </div>`;
  document.body.appendChild(menu);
  __openMenuEl = menu;
  const onDoc = (ev)=>{ if (!menu.contains(ev.target)) { closeOpenMenu(); document.removeEventListener('mousedown', onDoc); } };
  setTimeout(()=> document.addEventListener('mousedown', onDoc), 0);
  menu.querySelector('#mi_rename').onclick = ()=>{ closeOpenMenu(); beginInlineRename(btnRef, name); };
  menu.querySelector('#mi_delete').onclick = async ()=>{
    closeOpenMenu();
    const ok = confirm(`Delete collection "${name}"?\nThis will remove ALL related papers permanently.`);
    if (!ok) return;
    try {
      await apiFetch('/collections/delete', { method:'POST', body: JSON.stringify({ name, confirm: true }) });
      await listCollections();
      if (state.selectedCollection === name) state.selectedCollection = 'Default';
      saveState();
      renderCollections(state.collections);
      await listDocs();
    } catch (e) { alert('Delete failed'); }
  };
}

function showChatMenu(anchor, chatId, btnRef) {
  closeOpenMenu();
  const menu = ce('div');
  menu.className = 'context-menu';
  const rect = anchor.getBoundingClientRect();
  menu.style.left = (rect.left - 6) + 'px';
  menu.style.top = (rect.bottom + 6) + 'px';
  menu.innerHTML = `
    <div class="list">
      <div class="item" id="ci_rename">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none"><path d="M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z" stroke="currentColor" stroke-width="1.6"/></svg>
        <span>Rename</span>
      </div>
      <div class="item warn" id="ci_delete">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="none"><path d="M5 7h14M9 7V5h6v2m-8 0l1 12h8l1-12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
        <span>Delete</span>
      </div>
    </div>`;
  document.body.appendChild(menu); __openMenuEl = menu;
  const onDoc = (ev)=>{ if (!menu.contains(ev.target)) { closeOpenMenu(); document.removeEventListener('mousedown', onDoc); } };
  setTimeout(()=> document.addEventListener('mousedown', onDoc), 0);
  menu.querySelector('#ci_rename').onclick = ()=>{ closeOpenMenu(); beginInlineRenameChat(btnRef, chatId, btnRef.textContent || 'Chat'); };
  menu.querySelector('#ci_delete').onclick = ()=>{
    closeOpenMenu();
    const ok = confirm('Delete this chat?'); if (!ok) return;
    const idx = state.chats.findIndex(x=>x.id===chatId);
    if (idx>=0) {
      state.chats.splice(idx,1);
      if (state.activeChatId===chatId) state.activeChatId = state.chats[0]?.id || null;
      saveState(); renderSidebar(); renderChat();
    }
  };
}

function beginInlineRenameChat(btn, chatId, oldTitle) {
  const list = document.getElementById('chat_list');
  if (!list) return;
  const idxNode = btn;
  const input = ce('input', 'select __col_edit');
  input.value = oldTitle;
  input.style.width = btn.offsetWidth ? (btn.offsetWidth + 'px') : '100%';
  list.replaceChild(input, idxNode.parentElement || idxNode);
  input.focus();
  const cancel = ()=>{ try { list.replaceChild(idxNode.parentElement || idxNode, input); } catch(_){} };
  const commit = ()=>{
    const val = (input.value || '').trim();
    if (!val || val === oldTitle) { cancel(); return; }
    const chat = state.chats.find(x=>x.id===chatId);
    if (chat) { chat.title = val; saveState(); renderSidebar(); }
  };
  input.addEventListener('keydown', (e)=>{ if (e.key==='Enter') commit(); if (e.key==='Escape') cancel(); });
  input.addEventListener('blur', commit);
}

async function renderWatchList() {
  const list = document.getElementById('watch_list');
  if (!list) return;
  list.innerHTML = '';
  try {
    const s = await apiFetch('/settings');
    const v = s.settings || {};
    const map = Array.isArray(v.watch_map) ? v.watch_map : [];
    map.forEach(entry => {
      const row = ce('div', 'row');
      row.style.justifyContent = 'space-between';
      row.style.alignItems = 'center';
      row.style.padding = '8px 10px';
      row.style.border = '1px solid var(--muted)';
      row.style.borderRadius = '12px';
      row.style.marginBottom = '6px';
      const col = entry.collection || 'Default';
      row.innerHTML = `<div style="color:var(--text)">${entry.path}</div>
        <div class="row" style="gap:8px; align-items:center">
          <div class="pill" title="Collection">${col}</div>
          <div class="icon-chip warn" title="Remove" role="button" aria-label="Remove">
            <svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M5 7h14M9 7V5h6v2m-8 0l1 12h8l1-12' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg>
          </div>
        </div>`;
      const del = row.querySelector('.icon-chip.warn');
      del.onclick = async ()=>{
        try {
          const s2 = await apiFetch('/settings');
          const v2 = s2.settings || {};
          const arr = Array.isArray(v2.watch_map) ? v2.watch_map : [];
          const next = arr.filter(e => e.path !== entry.path);
          await apiFetch('/settings', { method:'POST', body: JSON.stringify({ watch_map: next }) });
          await renderWatchList();
        } catch(e) {}
      };
      list.appendChild(row);
    });
  } catch(e) { /* ignore */ }
}

async function restoreUploadJobs(signal, renderToken) {
  const list = document.getElementById('upload_list');
  if (!list) return;
  // clear existing to avoid duplicates during re-render
  console.log(`[DEBUG] restoreUploadJobs: clearing list with ${list.children.length} children`);
  list.innerHTML = '';
  try {
    const data = await apiFetch('/jobs', { signal });
    const jobs = data.jobs || {};
    console.log(`[DEBUG] restoreUploadJobs: fetched ${Object.keys(jobs).length} jobs`);
    let addedJobCount = 0;
    const currentCol = state.selectedCollection || 'Default';
    const seen = new Set();
    Object.entries(jobs).forEach(([jobId, st]) => {
      if (renderToken != null && renderToken !== __libRenderToken) return; // outdated render, stop adding
      // Only show in-progress jobs in the upload list; hide finished (success or error)
      if (st && st.done) return;
      // filter by selected collection if available on job state
      const jobCol = st && (st.collection || 'Default');
      if (jobCol !== currentCol) return;
      // Skip if we already have a row for this filename
      const row = ce('div', 'row');
      row.style.justifyContent = 'space-between';
      row.style.alignItems = 'center';
      row.style.padding = '8px 10px';
      row.style.border = '1px solid var(--muted)';
      row.style.borderRadius = '12px';
      row.style.marginBottom = '6px';
      const name = st.filename || st.paper_id || jobId;
      row.innerHTML = `<div style="color:var(--text)">${name}</div>
        <div class="row" style="gap:8px; align-items:center">
          <div class="icon-chip reprocess" title="Reprocess" role="button" aria-label="Reprocess">
            <svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M3 12a9 9 0 1 1 2.64 6.36' stroke='currentColor' stroke-width='1.6'/><path d='M3 16v-4h4' stroke='currentColor' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/></svg>
          </div>
          <div class="icon-chip status" title="Status">
            <svg class="ring" viewBox="0 0 36 36"><path class="track" d="M18 2a16 16 0 1 1 0 32 16 16 0 0 1 0-32"/><path class="arc" d="M18 2 a16 16 0 0 1 0 0"/></svg>
            <svg class="icon" viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><circle cx='12' cy='12' r='6' stroke='currentColor' stroke-width='1.6'/></svg>
          </div>
          <div class="icon-chip warn" title="Delete"><svg viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M5 7h14M9 7V5h6v2m-8 0l1 12h8l1-12' stroke='currentColor' stroke-width='1.6' stroke-linecap='round'/></svg></div>
        </div>`;
      if (renderToken != null && renderToken !== __libRenderToken) return; // outdated before append
      list.appendChild(row);
      addedJobCount++;
      console.log(`[DEBUG] Added job row ${addedJobCount}: ${name} (jobId: ${jobId})`);
      const statusChip = row.querySelector('.icon-chip.status');
      const repChip = row.querySelector('.icon-chip.reprocess');
      repChip.onclick = async ()=>{
        if (!st.paper_id) return; // cannot reprocess until paper_id known
        try { await apiFetch('/docs/reprocess', { method:'POST', body: JSON.stringify({ paper_id: st.paper_id }) }); } catch(e){}
      };
      // init progress
      const initPr = Math.max(0, Math.min(100, st.progress || 0));
      const arc = row.querySelector('.icon-chip.status .arc');
      if (arc) {
        const end = (initPr/100)*2*Math.PI; const r=16, cx=18, cy=18; const x=cx + r*Math.sin(end), y=cy - r*Math.cos(end); const large = end > Math.PI ? 1 : 0; arc.setAttribute('d', `M18 2 a16 16 0 ${large} 1 ${x-18} ${y-2}`); arc.setAttribute('stroke', initPr>=100 ? '#22c55e' : '#6ee7b7');
      }
      if (st.paper_id) { seen.add(st.paper_id); }
      {
        // Start polling to update
        const base = state.settings.api_base || '';
        const poll = async ()=>{
          try {
            const jr = await fetch((base||'') + API_BASE + '/jobs/' + jobId);
            const jd = await jr.json();
            if (renderToken != null && renderToken !== __libRenderToken) return; // stop polling on new render
            if (!jr.ok || !jd.success) return setTimeout(poll, 1200);
            const pr = Math.max(0, Math.min(100, jd.job.progress || 0));
            const arc2 = row.querySelector('.icon-chip.status .arc');
            if (arc2) { const end=(pr/100)*2*Math.PI; const r=16, cx=18, cy=18; const x=cx + r*Math.sin(end), y=cy - r*Math.cos(end); const large=end>Math.PI?1:0; arc2.setAttribute('d', `M18 2 a16 16 0 ${large} 1 ${x-18} ${y-2}`); arc2.setAttribute('stroke', pr>=100?'#22c55e':'#6ee7b7'); }
            if (!jd.job.done) return setTimeout(poll, 1000);
            // Remove finished job rows (success or error) from the in-progress list
            try { row.remove(); } catch(_) {}
            if (jd.job.success && jd.job.paper_id) seen.add(jd.job.paper_id);
            if (renderToken != null && renderToken !== __libRenderToken) return; // don't trigger refresh on outdated
          } catch(e) { setTimeout(poll, 1200); }
        };
        poll();
      }
    });
    console.log(`[DEBUG] restoreUploadJobs: added ${addedJobCount} job rows total`);
    return seen;
  } catch(e) { 
    console.log(`[DEBUG] restoreUploadJobs error:`, e);
    return new Set(); 
  }
}

async function uploadFiles(files) {
  const total = files.length;
  let done = 0;
  for (const f of files) {
    const fd = new FormData(); fd.append('file', f);
    const base = state.settings.api_base || '';
    const res = await fetch((base||'') + API_BASE + '/upload', { method:'POST', body: fd });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error_message || 'Upload failed');
    done += 1;
    const pct = Math.round((done/total)*100);
    setUploadProgress(pct);
  }
}

function renderRoute() {
  document.querySelectorAll('.nav a').forEach(a=>a.classList.remove('active'));
  if (state.route === '#/library') {
    $('#nav_library').classList.add('active');
  } else if (state.route === '#/chat' || state.route === '' || state.route === '#' || state.route == null) {
    $('#nav_chat').classList.add('active');
  }
  if (state.route === '#/library') renderLibrary();
  else if (state.route === '#/settings') renderSettingsPage();
  else renderChatPage();
}

function renderSidebar() {
  const list = $('#chat_list');
  list.innerHTML = '';
  // Show newest chats first
  state.chats
    .slice()
    .sort((a,b)=> (b.created_at||0) - (a.created_at||0))
    .forEach((c) => {
      const btn = ce('button', 'btn');
      btn.textContent = c.title || 'Chat';
      if (state.activeChatId === c.id) btn.classList.add('btn-active');
      btn.onclick = () => { state.activeChatId = c.id; saveState(); renderSidebar(); renderChat(); };

      // Wrap for kebab menu
      const wrap = ce('div');
      wrap.style.position = 'relative';
      wrap.appendChild(btn);

      const dots = ce('div');
      dots.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><circle cx="5" cy="12" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="19" cy="12" r="2"/></svg>';
      dots.style.position = 'absolute';
      dots.style.right = '12px';
      dots.style.top = '50%';
      dots.style.transform = 'translateY(-50%)';
      dots.style.display = 'none';
      dots.style.cursor = 'pointer';
      dots.style.width = '20px';
      dots.style.height = '20px';
      wrap.onmouseenter = ()=>{ dots.style.display = ''; };
      wrap.onmouseleave = ()=>{ dots.style.display = 'none'; };
      dots.onclick = (e)=>{ e.stopPropagation(); showChatMenu(dots, c.id, btn); };
      wrap.appendChild(dots);

      list.appendChild(wrap);
    });
}

function renderChat(showTyping=false) {
  const chatEl = $('#chat');
  chatEl.innerHTML = '';
  const chat = getActiveChat();
  if (!chat) return;
  chat.history.forEach((turn, idx) => {
    const u = ce('div', 'msg user');
    const userAvatar = getUserAvatarHtml();
    // User bubble on the right: bubble first, avatar at far right
    u.innerHTML = `<div class="bubble">${escapeHtml(turn.user)}</div>${userAvatar}`;
    chatEl.appendChild(u);
    const isLast = idx === chat.history.length - 1;
    const aiText = turn.ai || '';
    if (!(showTyping && isLast && aiText.trim() === '')) {
      const a = ce('div', 'msg ai');
      const html = renderMarkdown(aiText);
      const aiAvatar = `<div class="avatar ai-avatar" style="background-image:url('/static/logo.png?v=1')"></div>`;
      a.innerHTML = `${aiAvatar}<div class="bubble"><div class="md">${html}</div></div>`;
      chatEl.appendChild(a);
    }
  });
  if (showTyping) {
    const t = ce('div', 'msg ai');
    const aiAvatar = `<div class="avatar ai-avatar" style="background-image:url('/static/logo.png?v=1')"></div>`;
    t.innerHTML = `${aiAvatar}<div class="bubble"><span class="typing" aria-live="polite" aria-label="AI is thinking">
      <span class="dot"></span><span class="dot"></span><span class="dot"></span>
    </span></div>`;
    chatEl.appendChild(t);
  }
  chatEl.scrollTop = chatEl.scrollHeight;
}

function renderMenu(menu) {
  const menuEl = $('#menu');
  menuEl.innerHTML = '';
  const row = ce('div');
  menu.forEach((opt, idx) => {
    const b = ce('button');
    b.textContent = `${idx + 1}. ${opt}`;
    b.onclick = () => sendMenuChoice(idx + 1);
    row.appendChild(b);
  });
  menuEl.appendChild(row);
}

function renderDocsPanel() {
  const docs = $('#docs');
  docs.innerHTML = '';
  state.docs.forEach(d => {
    const btn = ce('button');
    btn.textContent = `${d.title} (${d.year})`;
    docs.appendChild(btn);
  });
}

function renderRefsPanel() {
  const list = document.getElementById('refs');
  if (!list) return;
  list.innerHTML = '';
  const chat = getActiveChat();
  const titles = (chat && chat.bibliography) ? chat.bibliography : [];
  titles.forEach(t => {
    const item = ce('div', 'item');
    item.textContent = t;
    list.appendChild(item);
  });
}

function setUploadProgress(pct) {
  const btn = document.getElementById('btn_upload_lib');
  if (!btn) return;
  if (pct > 0 && pct < 100) {
    btn.style.setProperty('--progress', pct + '%');
    btn.disabled = true;
  } else if (pct >= 100) {
    btn.style.setProperty('--progress', '100%');
    btn.disabled = false;
    // brief success state
  } else {
    btn.style.setProperty('--progress', '0%');
    btn.disabled = false;
  }
}

async function onUploadFile(e) {
  const file = e.target.files?.[0];
  if (!file) return;
  try {
    const fd = new FormData();
    fd.append('file', file);
    const base = state.settings.api_base || '';
    const res = await fetch((base || '') + API_BASE + '/upload', {
      method: 'POST',
      body: fd,
    });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.error_message || 'Upload failed');
    await listDocs();
  } catch (err) {
    console.error('Upload failed', err);
  } finally {
    e.target.value = '';
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"]+/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function renderMarkdown(src) {
  try {
    if (window.marked && typeof window.marked.parse === 'function') {
      // Security: basic sanitize by escaping script tags before feeding to marked
      const safe = (src || '').replace(/<\s*script/gi, '&lt;script');
      return window.marked.parse(safe);
    }
  } catch (_) {}
  // Fallback to escaped plain text
  return (src || '').split('\n').map(ln => `<p>${escapeHtml(ln)}</p>`).join('');
}

async function init() {
  if (!state.chats.length) newChat('My first chat');
  if (!state.activeChatId && state.chats.length) state.activeChatId = state.chats[0].id;
  renderRoute();
  listDocs();
  // Settings modal
  $('#btn_settings').onclick = ()=> { location.hash = '#/settings'; };
  // Replace top-left favicon in settings modal header to use new logo if shown by OS previews
  $('#settings_close').onclick = ()=> toggleModal('settings_modal', false);
  $('#settings_save').onclick = async ()=> {
    state.settings.api_base = $('#setting_api').value.trim();
    const keyInputVal = ($('#openai_key').value || '').trim();
    if (keyInputVal && keyInputVal !== '********') {
      state.settings.openai_key = keyInputVal;
    }
    // watch settings
    const payload = {
      api_base: state.settings.api_base,
      openai_key: state.settings.openai_key,
      watch_enabled: $('#watch_enabled').checked,
      watch_interval_seconds: parseInt($('#watch_interval').value || '300', 10),
      watch_directories: ($('#watch_dirs').value || '').split(',').map(s=>s.trim()).filter(Boolean),
    };
    try {
      // Only include key if really provided (ignore masked)
      if (state.settings.openai_key && state.settings.openai_key !== '********') {
        payload.openai_key = state.settings.openai_key;
      }
      await apiFetch('/settings', { method: 'POST', body: JSON.stringify(payload) });
    } catch (e) { console.error('Save settings failed', e); }
    saveState(); toggleModal('settings_modal', false);
  };
  $('#watch_scan_now').onclick = async ()=> { try { await apiFetch('/watch/scan-now', { method: 'POST' }); } catch (e) { console.error(e); } };
  $('#btn_account').onclick = ()=> toggleModal('account_modal', true);
  $('#account_close').onclick = ()=> toggleModal('account_modal', false);
  $('#account_save').onclick = ()=> { toggleModal('account_modal', false); };
  // Prefill settings
  $('#setting_api').value = state.settings.api_base || '';
  $('#openai_key').value = state.settings.openai_key || '';
  // Prefill watch settings from server
  try {
    const s = await apiFetch('/settings');
    const v = s.settings || {};
    const we = document.getElementById('watch_enabled');
    const wi = document.getElementById('watch_interval');
    const wd = document.getElementById('watch_dirs');
    if (we) we.checked = !!v.watch_enabled;
    if (wi) wi.value = v.watch_interval_seconds || 300;
    if (wd) wd.value = (v.watch_directories || []).join(',');
    const okEl = document.getElementById('openai_key');
    if (okEl && (v.openai_key_present || v.openai_key_preview)) {
      okEl.value = v.openai_key_preview || '********';
    }
  } catch (e) {
    // ignore
  }
  window.addEventListener('hashchange', ()=>{ state.route = location.hash || '#/chat'; renderRoute(); });
}

function toggleModal(id, open) { const el = document.getElementById(id); if (!el) return; el.classList.toggle('hidden', !open); }

init();


// Settings single-page view (global app settings)
function renderSettingsPage() {
  const view = $('#view');
  view.innerHTML = '';
  const wrap = ce('div', 'container-library');
  // Left: sections list (tabs-like)
  const left = ce('div', 'card left-pane');
  left.innerHTML = `
    <div class="header">Settings</div>
    <div class="body list">
      <div class="item active" id="tab_general">General</div>
      <div class="item" id="tab_reset">Reset</div>
    </div>`;
  wrap.appendChild(left);

  // Right: content panel (switchable)
  const right = ce('div', 'card center-pane');
  const generalHtml = `
    <div class="header">General</div>
    <div class="body settings-pane" style="display:flex; flex-direction:column; gap:14px">
      <div class="form-row">
        <label>API Base URL</label>
        <input id="set_api_base" class="select" placeholder="http://localhost:31415" />
      </div>
      <div class="form-row">
        <label>OpenAI API Key</label>
        <div class="input-inline">
          <input id="set_openai_key" class="select" type="password" placeholder="sk-..." />
          <button id="toggle_key_visibility" class="eye-btn" title="Show/Hide" aria-label="Show/Hide">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M1 12s4-7 11-7 11 7 11 7-4 7-11 7S1 12 1 12Z" stroke="currentColor" stroke-width="1.6"/><circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.6"/></svg>
          </button>
        </div>
      </div>
      <div class="form-row">
        <label>Embedding model (Global)</label>
        <select id="set_embedding_model" class="select"></select>
      </div>
      <div id="embed_lock_hint" style="color:var(--sub); font-size:12px; margin-top:-6px"></div>
      <div class="row" style="justify-content:flex-start">
        <div class="save-container" style="width:100%">
          <button id="settings_save2" class="primary" style="width:100%">Save</button>
        </div>
      </div>
    </div>`;
  const resetHtml = `
    <div class="header">Reset</div>
    <div class="body" style="display:flex; flex-direction:column; gap:12px">
      <div class="input-plain" style="white-space:normal">Initialize database and delete <b>ALL</b> data. Only the Default collection will remain. This action cannot be undone.</div>
      <div><button id="btn_reset_confirm" class="primary" style="max-width:240px">Initialize database</button></div>
    </div>`;
  right.innerHTML = generalHtml;
  wrap.appendChild(right);
  view.appendChild(wrap);

  // Prefill
  (async ()=>{
    try {
      const s = await apiFetch('/settings');
      const v = s.settings || {};
      // Fill from .env-backed values if present
      $('#set_api_base').value = v.api_base || state.settings.api_base || '';
      if (v.openai_key_present || v.openai_key_preview) {
        // use a longer masked preview to avoid implying short keys
        const mask = '****************';
        const el = document.getElementById('set_openai_key');
        el.value = mask;
        el.dataset.masked = '1';
      } else {
        const el = document.getElementById('set_openai_key');
        el.value = state.settings.openai_key || '';
        el.dataset.masked = '0';
      }
      // Populate embedding model options (OpenAI + Ollama popular)
      const modelSelect = document.getElementById('set_embedding_model');
      const options = [
        // OpenAI
        { value: 'openai:text-embedding-3-small', label: 'OpenAI: text-embedding-3-small (1536d)' },
        { value: 'openai:text-embedding-3-large', label: 'OpenAI: text-embedding-3-large (3072d)' },
        // Ollama popular community models
        { value: 'ollama:nomic-embed-text', label: 'Ollama: nomic-embed-text' },
        { value: 'ollama:bge-m3', label: 'Ollama: bge-m3' },
        { value: 'ollama:gte-base', label: 'Ollama: gte-base' },
        { value: 'ollama:gte-large', label: 'Ollama: gte-large' },
        { value: 'ollama:all-minilm-l6-v2', label: 'Ollama: all-MiniLM-L6-v2' }
      ];
      modelSelect.innerHTML = '';
      options.forEach(o=>{ const op = document.createElement('option'); op.value=o.value; op.textContent=o.label; modelSelect.appendChild(op); });
      // Load from model_config.json via API if present
      try {
        const emb = await apiFetch('/config/embedding');
        const pv = emb.provider, mv = emb.model;
        if (pv && mv) modelSelect.value = `${pv}:${mv}`;
      } catch(e) {}
      const currentModel = modelSelect.value || v.embedding_model || 'openai:text-embedding-3-small';
      modelSelect.value = currentModel;
      if (v.embedding_provider_locked || v.embedding_model_locked) {
        modelSelect.disabled = true;
        $('#embed_lock_hint').textContent = `Locked by existing data: ${currentModel}`;
      }
    } catch(e){/* ignore */}
  })();

  // Save with model-change confirmation (General tab)
  $('#settings_save2').onclick = async ()=>{
    try {
      const s = await apiFetch('/settings');
      const v = s.settings || {};
      const next = {
        api_base: $('#set_api_base').value.trim(),
        openai_key: $('#set_openai_key').value.trim(),
      };
      const currentModel = v.embedding_model || '';
      const desiredModel = $('#set_embedding_model').value;
      if ((v.embedding_provider_locked || v.embedding_model_locked) && desiredModel !== currentModel) {
        const ok = confirm(`Changing embedding model to "${desiredModel}" will require rebuilding ALL embeddings and may clear the index. Continue?`);
        if (!ok) return;
        next.embedding_model = desiredModel;
        next.embedding_rebuild = true;
      } else if (!v.embedding_provider_locked && !v.embedding_model_locked && desiredModel) {
        next.embedding_model = desiredModel;
      }
      // Persist to settings for runtime and to model_config.json for global reference
      await apiFetch('/settings', { method:'POST', body: JSON.stringify(next) });
      const [prov, mod] = (desiredModel || currentModel).split(':');
      if (prov && mod) { await apiFetch('/config/embedding', { method:'POST', body: JSON.stringify({ provider: prov, model: mod }) }); }
      alert('Saved.');
    } catch(e) { console.error(e); alert('Save failed'); }
  };

  // Toggle eye visibility
  const eye = document.getElementById('toggle_key_visibility');
  let keyInput = document.getElementById('set_openai_key');
  if (eye && keyInput) {
    eye.onclick = async ()=> {
      const parent = keyInput.parentElement;
      const isMasked = keyInput.dataset.masked === '1';
      const plainId = 'openai_key_plain_view';
      if (isMasked) {
        try {
          // fetch real key
          const res = await fetch(API_BASE + '/secret/openai-key');
          const data = await res.json();
          if (!res.ok || !data.success || !data.key) throw new Error('no key');
          // hide input and show a plain div bubble
          keyInput.style.display = 'none';
          let plain = document.getElementById(plainId);
          if (!plain) {
            plain = document.createElement('div');
            plain.id = plainId;
            plain.className = 'input-plain';
            parent.insertBefore(plain, eye); // before eye button
          }
          plain.textContent = data.key;
          keyInput.dataset.masked = '0';
        } catch(e) {
          alert('Cannot display key (server disabled or not set).');
        }
      } else {
        // hide plain view and restore masked input
        const plain = document.getElementById(plainId);
        if (plain) plain.remove();
        keyInput.style.display = '';
        keyInput.type = 'password';
        keyInput.value = '****************';
        keyInput.dataset.masked = '1';
      }
    };
  }

  // side tabs switching
  const panel = right;
  const toGeneral = ()=>{ document.getElementById('tab_general').classList.add('active'); document.getElementById('tab_reset').classList.remove('active'); panel.innerHTML = generalHtml; rewireGeneral(); };
  const toReset = ()=>{ document.getElementById('tab_reset').classList.add('active'); document.getElementById('tab_general').classList.remove('active'); panel.innerHTML = resetHtml; wireReset(); };
  document.getElementById('tab_general').onclick = toGeneral;
  document.getElementById('tab_reset').onclick = toReset;

  function rewireGeneral(){
    // re-bind handlers after swapping innerHTML
    document.getElementById('settings_save2').onclick = async ()=>{
      try {
        const s = await apiFetch('/settings');
        const v = s.settings || {};
        const next = { api_base: document.getElementById('set_api_base').value.trim(), openai_key: document.getElementById('set_openai_key').value.trim() };
        const currentModel = v.embedding_model || '';
        const desiredModel = document.getElementById('set_embedding_model').value;
        if ((v.embedding_provider_locked || v.embedding_model_locked) && desiredModel !== currentModel) {
          const ok = confirm(`Changing embedding model to "${desiredModel}" will require rebuilding ALL embeddings and may clear the index. Continue?`);
          if (!ok) return;
          next.embedding_model = desiredModel; next.embedding_rebuild = true;
        } else if (!v.embedding_provider_locked && !v.embedding_model_locked && desiredModel) { next.embedding_model = desiredModel; }
        await apiFetch('/settings', { method:'POST', body: JSON.stringify(next) });
        const [prov, mod] = (desiredModel || currentModel).split(':'); if (prov && mod) await apiFetch('/config/embedding', { method:'POST', body: JSON.stringify({ provider: prov, model: mod }) });
        alert('Saved.');
      } catch(e){ alert('Save failed'); }
    };
    const eye2 = document.getElementById('toggle_key_visibility');
    const key2 = document.getElementById('set_openai_key');
    if (eye2 && key2) eye2.onclick = ()=>{ key2.type = key2.type === 'password' ? 'text' : 'password'; };
  }
  function wireReset(){
    const btn = document.getElementById('btn_reset_confirm');
    if (!btn) return;
    btn.onclick = async ()=>{
      const ok = confirm('Initialize database and delete ALL data?\n\nThis action is irreversible. All papers, jobs, and files will be removed.');
      if (!ok) return;
      const base = state.settings.api_base || '';
      const headers = { 'Content-Type': 'application/json' };
      const attempts = [
        { url: (base||'') + API_BASE + '/admin/init', opts: { method:'POST', headers, body: JSON.stringify({ confirm: true }) } },
        { url: (base||'') + API_BASE + '/reset', opts: { method:'POST', headers, body: JSON.stringify({ confirm: true }) } },
        { url: (base||'') + API_BASE + '/admin/init', opts: { method:'GET' } },
        { url: (base||'') + '/admin/init', opts: { method:'POST', headers, body: JSON.stringify({ confirm: true }) } },
        { url: (base||'') + '/api/reset', opts: { method:'POST', headers, body: JSON.stringify({ confirm: true }) } },
        { url: (base||'') + '/admin/init', opts: { method:'GET' } },
      ];
      let lastErr = null;
      try {
        for (const a of attempts) {
          try {
            const r = await fetch(a.url, a.opts);
            if (r.ok) { await r.json().catch(()=>({})); lastErr = null; break; }
            lastErr = await r.text();
          } catch (e) { lastErr = e?.message || String(e); }
        }
        if (lastErr) throw new Error(typeof lastErr === 'string' ? lastErr : JSON.stringify(lastErr));
        alert('Reset done.');
        await listCollections();
        state.selectedCollection='Default';
        saveState();
      } catch(e){
        console.error('Reset failed', e);
        alert('Reset failed: ' + (e && e.message ? e.message : 'unknown'));
      }
    };
  }
}

// Admin page: Reset/Initialize
function renderAdminPage() {
  const view = $('#view');
  view.innerHTML = '';
  const wrap = ce('div', 'container-library');
  const left = ce('div', 'card left-pane');
  left.innerHTML = `
    <div class="header">Admin</div>
    <div class="body list">
      <div class="item active">Reset</div>
    </div>`;
  wrap.appendChild(left);

  const center = ce('div', 'card center-pane');
  center.innerHTML = `
    <div class="header">Reset workspace</div>
    <div class="body" style="display:flex; flex-direction:column; gap:12px">
      <div class="input-plain">This will delete all documents, jobs and local files. Only the Default collection will remain.</div>
      <div class="row" style="gap:10px">
        <button id="btn_reset_now" class="primary" style="max-width:220px">Initialize database</button>
      </div>
    </div>`;
  wrap.appendChild(center);
  view.appendChild(wrap);

  document.getElementById('btn_reset_now').onclick = async ()=>{
    const ok = confirm('Initialize database and delete ALL data?\n\nThis action is irreversible. All papers, jobs, and files will be removed.');
    if (!ok) return;
    try {
      const res = await apiFetch('/admin/init', { method:'POST', body: JSON.stringify({ confirm: true }) });
      alert('Reset done. You may refresh Library.');
      // refresh local UI caches
      await listCollections();
      state.selectedCollection = 'Default';
      saveState();
      if (location.hash !== '#/library') location.hash = '#/library';
      else renderLibrary();
    } catch(e) {
      alert('Reset failed');
    }
  };
}
