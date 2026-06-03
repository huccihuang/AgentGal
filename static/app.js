marked.setOptions({ breaks: true, gfm: true });

document.addEventListener("alpine:init", () => {
  Alpine.data("agentGalApp", () => ({
    busy: false,
    observeMode: false,
    characterCount: 0,
    consolidating: false,
    consolidatingPollTimer: null,
    consolidatingPollGen: 0,
    noticeText: "",
    noticeTone: "",
    noticeTimer: null,
    storyModalOpen: false,
    confirmDialog: {
      open: false,
      title: "",
      body: "",
      confirmText: "确定",
    },
    confirmAction: null,
    ...window.agentGalMemoryGraph.createState(),
    hasSave: false,
    isCompact: false,
    drawerOpen: false,
    mobileMenuOpen: false,
    mmSection: null,
    inputText: "",
    stories: [],
    messages: [],
    choices: [],
    choiceStatus: "hidden",
    saves: [],
    worlds: [],
    activeWorldId: "",
    currentSaveId: "",
    currentStoryId: "",
    worldlineOpen: false,
    worldlineZoom: 1,
    worldlineDrag: null,
    isSaving: false,
    characters: [],
    charactersLoading: false,
    _fetchCharactersSeq: 0,
    saveLoading: false,
    saveError: "",
    initialRecent: [],
    initialChoices: [],
    nextMessageId: 1,
    oldestLoadedTurn: null,
    historyExhausted: false,
    loadingOlder: false,
    gameDates: [],
    gameDatesLoaded: false,
    gameDatesLoading: false,
    dateMenuOpen: false,
    searchQuery: "",
    searchResults: [],
    searchLoading: false,
    searchTimer: null,
    historySearchOpen: false,
    nearLatest: true,
    unreadNewMessages: false,
    activeStreamController: null,
    activeStreamId: 0,
    toastTimers: new Map(),
    toasts: [],
    mediaQuery: null,
    visualViewport: null,
    viewportBaseline: 0,
    keyboardVisible: false,
    inputFocused: false,
    handleViewportChangeBound: null,
    handleVisualViewportChangeBound: null,
    agentStyleByAuthor: {},
    agentPalette: [
      { color: "#b45a64", border: "rgba(180, 90, 100, 0.18)", background: "rgba(180, 90, 100, 0.08)" },
      { color: "#6a8165", border: "rgba(106, 129, 101, 0.2)", background: "rgba(106, 129, 101, 0.08)" },
      { color: "#5b7d86", border: "rgba(91, 125, 134, 0.2)", background: "rgba(91, 125, 134, 0.08)" },
      { color: "#95684d", border: "rgba(149, 104, 77, 0.2)", background: "rgba(149, 104, 77, 0.08)" },
    ],

    async init() {
      if ("scrollRestoration" in window.history) {
        window.history.scrollRestoration = "manual";
      }
      this.setupViewportListener();
      this.setupVisualViewportListener();

      try {
        const storiesResponse = await this.fetchJson("/api/stories");
        this.stories = storiesResponse.stories || [];
      } catch (error) {
        this.setNotice("故事列表加载失败，稍后仍可重新尝试。", "error", true);
      }

      try {
        const initState = await this.fetchJson("/api/init");
        this.hasSave = Boolean(initState.has_save);
        this.characterCount = initState.character_count || 0;
        this.initialRecent = initState.recent || [];
        this.initialChoices = initState.last_choices || [];
        if (this.hasSave) {
          this.continueGame(this.initialRecent, this.initialChoices, { silent: true });
        } else {
          this.storyModalOpen = true;
        }
      } catch (error) {
        this.setNotice("初始化失败，部分功能可能暂时不可用。", "error", true);
        this.storyModalOpen = true;
      }

      await this.refreshSaves({ quiet: true });
      await this.fetchCharacters();
    },

    destroy() {
      if (this.mediaQuery && this.handleViewportChangeBound) {
        this.mediaQuery.removeEventListener("change", this.handleViewportChangeBound);
      }
      if (this.visualViewport && this.handleVisualViewportChangeBound) {
        this.visualViewport.removeEventListener("resize", this.handleVisualViewportChangeBound);
      }
      this.destroyMemoryGraphNetwork();
      this.cancelActiveStream({ silent: true });
      this.stopConsolidationPolling();
      if (this.noticeTimer) {
        window.clearTimeout(this.noticeTimer);
      }
      this.toastTimers.forEach(timeoutId => window.clearTimeout(timeoutId));
      this.toastTimers.clear();
    },

    setupViewportListener() {
      this.mediaQuery = window.matchMedia("(max-width: 920px)");
      this.handleViewportChangeBound = (event) => {
        this.isCompact = event.matches;
        this.drawerOpen = !event.matches;
        if (!event.matches) {
          this.inputFocused = false;
          this.mobileMenuOpen = false;
          this.mmSection = null;
        }
        this.updateKeyboardState();
        this.$nextTick(() => this.resizeComposer());
      };
      this.handleViewportChangeBound(this.mediaQuery);
      this.mediaQuery.addEventListener("change", this.handleViewportChangeBound);
    },

    setupVisualViewportListener() {
      if (!window.visualViewport) return;
      this.visualViewport = window.visualViewport;
      this.handleVisualViewportChangeBound = () => this.updateKeyboardState();
      this.updateKeyboardState();
      this.visualViewport.addEventListener("resize", this.handleVisualViewportChangeBound);
    },

    updateKeyboardState() {
      const viewportHeight = this.visualViewport ? this.visualViewport.height : window.innerHeight;
      if (!viewportHeight) return;
      if (!this.isCompact) {
        this.viewportBaseline = viewportHeight;
        this.keyboardVisible = false;
        return;
      }
      if (!this.viewportBaseline || viewportHeight > this.viewportBaseline) {
        this.viewportBaseline = viewportHeight;
      }
      this.keyboardVisible = this.inputFocused && (this.viewportBaseline - viewportHeight > 120);
    },

    handleEscape() {
      if (this.confirmDialog.open) {
        this.closeConfirm();
        return;
      }
      if (this.memoryGraphOpen) {
        this.closeMemoryGraph();
        return;
      }
      if (this.worldlineOpen) {
        this.closeWorldline();
        return;
      }
      if (this.historySearchOpen) {
        this.closeHistorySearch();
        return;
      }
      if (this.storyModalOpen && this.canDismissStoryModal()) {
        this.closeStoryModal();
        return;
      }
      if (this.drawerOpen && this.isCompact) {
        this.closeDrawer();
      }
    },

    noticeBadge() {
      if (this.noticeTone === "error") return "!";
      if (this.noticeTone === "success") return "OK";
      return "i";
    },

    isInputMode() {
      return this.isCompact && (this.inputFocused || this.keyboardVisible);
    },

    shouldShowNotice() {
      if (!this.noticeText) return false;
      return !this.isInputMode() || this.noticeTone === "error";
    },

    shouldShowChoices() {
      return this.choiceStatus !== "hidden" && !this.isInputMode();
    },

    choiceRailSubtitle() {
      if (this.choiceStatus === "loading") return "选项生成中，也可以直接输入你想说的话。";
      if (this.choiceStatus === "empty") return "这一轮没有预设选项。";
      return "选项只是建议，也可以直接输入你想说的话。";
    },

    trimmedInput() {
      return String(this.inputText || "").trim();
    },

    canDismissStoryModal() {
      return this.messages.length > 0;
    },

    storyCardClass(index) {
      return index % 2 === 0 ? "choice-card warm" : "choice-card cool";
    },

    storyCardCue(index) {
      if (index === 0) return "主线开场";
      return "另一种氛围";
    },

    toggleDrawer() {
      if (this.isCompact) {
        this.blurComposer();
      }
      this.drawerOpen = !this.drawerOpen;
    },

    closeDrawer() {
      this.drawerOpen = false;
    },

    toggleMobileMenu() {
      this.mobileMenuOpen = !this.mobileMenuOpen;
      if (!this.mobileMenuOpen) {
        this.mmSection = null;
        this.clearHistorySearch();
      }
    },

    closeMobileMenu() {
      this.mobileMenuOpen = false;
      this.mmSection = null;
    },

    async createSave() {
      await this.saveGame();
    },

    async openWorldline() {
      if (this.isCompact) {
        this.closeDrawer();
      }
      this.worldlineOpen = true;
      if (!this.worlds.length && !this.saveLoading) {
        await this.refreshSaves({ quiet: true });
      }
      this.ensureActiveWorld();
    },

    closeWorldline() {
      this.worldlineOpen = false;
    },

    selectWorld(storyId) {
      this.activeWorldId = storyId || "";
      this.worldlineZoom = 1;
      this.$nextTick(() => {
        const target = this.$refs?.worldlineScroll;
        if (!target) return;
        target.scrollLeft = 0;
        target.scrollTop = 0;
      });
    },

    resetFromDrawer() {
      this.closeDrawer();
      this.showResetConfirmation();
    },

    async fetchJson(url, options = {}) {
      const response = await fetch(url, options);
      if (!response.ok) {
        let detail = "";
        try {
          const text = await response.text();
          detail = text;
          if (text) {
            try {
              const parsed = JSON.parse(text);
              if (parsed && typeof parsed === "object") {
                detail = parsed.detail || parsed.error || text;
              }
            } catch (error) {
              detail = text;
            }
          }
        } catch (error) {
          detail = "";
        }
        throw new Error(`${url} failed: ${response.status}${detail ? ` ${detail.slice(0, 120)}` : ""}`);
      }
      return response.json();
    },

    deleteJson(url) {
      return this.fetchJson(url, { method: "DELETE" });
    },

    postJson(url, body) {
      return this.fetchJson(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    },

    clearNotice() {
      this.noticeText = "";
      this.noticeTone = "";
      if (this.noticeTimer) {
        window.clearTimeout(this.noticeTimer);
        this.noticeTimer = null;
      }
    },

    setNotice(text, tone = "success", sticky = false) {
      this.noticeText = text;
      this.noticeTone = tone;
      if (this.noticeTimer) {
        window.clearTimeout(this.noticeTimer);
        this.noticeTimer = null;
      }
      if (!sticky) {
        this.noticeTimer = window.setTimeout(() => {
          this.noticeText = "";
          this.noticeTone = "";
          this.noticeTimer = null;
        }, 3600);
      }
    },

    pushToast(message, tone = "success") {
      const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
      this.toasts.push({ id, message, tone });
      const timeoutId = window.setTimeout(() => {
        this.dismissToast(id);
      }, 3200);
      this.toastTimers.set(id, timeoutId);
    },

    dismissToast(id) {
      this.toasts = this.toasts.filter(toast => toast.id !== id);
      const timeoutId = this.toastTimers.get(id);
      if (timeoutId) {
        window.clearTimeout(timeoutId);
        this.toastTimers.delete(id);
      }
    },

    openConfirm({ title, body, confirmText = "确定", onConfirm }) {
      this.confirmDialog = {
        open: true,
        title,
        body,
        confirmText,
      };
      this.confirmAction = onConfirm;
    },

    closeConfirm() {
      this.confirmDialog = {
        open: false,
        title: "",
        body: "",
        confirmText: "确定",
      };
      this.confirmAction = null;
    },

    async runConfirm() {
      const action = this.confirmAction;
      this.closeConfirm();
      if (action) {
        await action();
      }
    },

    closeStoryModal() {
      if (!this.canDismissStoryModal()) return;
      this.storyModalOpen = false;
    },

    showResetConfirmation() {
      if (this.busy) {
        this.setNotice("请等待当前回合完成后再重开。", "error");
        return;
      }
      if (this.consolidating) {
        this.setNotice("记忆整理进行中，请稍后再重开。", "error");
        return;
      }
      this.openConfirm({
        title: "重新选择故事？",
        body: "当前运行时数据会被新开局覆盖。确认后会回到故事选择界面。",
        confirmText: "继续",
        onConfirm: async () => {
          this.storyModalOpen = true;
        },
      });
    },

    recentPreview() {
      const recent = [...this.initialRecent].reverse().find(message => String(message.content || "").trim());
      if (!recent) return "发现一个可继续的进度，读取后会恢复最近对话和建议行动。";
      return this.excerptPlain(recent.content, 88);
    },

    excerptPlain(content, maxLength = 72) {
      const plain = this.stripMarkdown(content || "").replace(/\s+/g, " ").trim();
      if (plain.length <= maxLength) return plain;
      return `${plain.slice(0, maxLength - 1)}…`;
    },

    stripMarkdown(value) {
      return String(value || "")
        .replace(/!\[[^\]]*]\([^)]*\)/g, "")
        .replace(/\[([^\]]+)]\([^)]*\)/g, "$1")
        .replace(/[`*_>#~-]/g, "")
        .replace(/\s+/g, " ");
    },

    handleComposerInput(event) {
      this.resizeComposer(event.target);
    },

    handleComposerFocus() {
      this.inputFocused = true;
      this.updateKeyboardState();
    },

    handleComposerBlur() {
      this.inputFocused = false;
      this.keyboardVisible = false;
    },

    blurComposer() {
      const textarea = this.$refs?.composerInput;
      if (textarea) {
        textarea.blur();
      }
    },

    resizeComposer(target = null) {
      const textarea = target || this.$refs?.composerInput;
      if (!textarea) return;
      const baseHeight = this.isCompact ? 52 : 108;
      const maxHeight = this.isCompact ? 136 : 220;
      textarea.style.height = `${baseHeight}px`;
      if (!textarea.value) return;
      textarea.style.height = `${Math.min(Math.max(textarea.scrollHeight, baseHeight), maxHeight)}px`;
    },

    resetComposer() {
      this.inputText = "";
      this.$nextTick(() => this.resizeComposer());
    },

    parseNarratorLine(rawLine) {
      const line = String(rawLine || "").replace(/\*\*/g, "").trim();
      const chineseSeparator = line.indexOf("：");
      const separatorIndex = chineseSeparator >= 0 ? chineseSeparator : line.indexOf(":");
      if (separatorIndex <= 0) return { key: "", value: "" };
      return { key: line.slice(0, separatorIndex).trim(), value: line.slice(separatorIndex + 1).trim() };
    },

    normalizeNarratorText(content) {
      const lines = String(content || "").split("\n");
      const normalized = [];
      const hiddenMetaKeys = new Set(["时间", "地点", "场景"]);

      for (const rawLine of lines) {
        const line = rawLine.trimEnd();
        const trimmed = line.trim();

        if (!trimmed) {
          normalized.push("");
          continue;
        }

        if (trimmed === "旁白" || trimmed === "**旁白**") {
          continue;
        }

        const { key: metaKey, value: metaValue } = this.parseNarratorLine(line);
        if (hiddenMetaKeys.has(metaKey)) {
          continue;
        }
        if (metaKey === "在场") {
          if (metaValue) normalized.push(metaValue);
          continue;
        }

        normalized.push(line);
      }

      return normalized.join("\n").replace(/\n{3,}/g, "\n\n").replace(/^\n+/, "");
    },

    isSafeLink(href) {
      const value = String(href || "").trim().toLowerCase();
      return value.startsWith("http://") || value.startsWith("https://") || value.startsWith("mailto:");
    },

    sanitizeHtml(html) {
      const template = document.createElement("template");
      template.innerHTML = html;

      const allowedTags = new Set([
        "a",
        "blockquote",
        "br",
        "code",
        "em",
        "h1",
        "h2",
        "h3",
        "hr",
        "li",
        "ol",
        "p",
        "pre",
        "strong",
        "ul",
      ]);
      const dropTags = new Set(["script", "style", "iframe", "object", "embed", "form", "input", "button", "textarea"]);
      const nodes = Array.from(template.content.querySelectorAll("*")).reverse();

      nodes.forEach(node => {
        const tag = node.tagName.toLowerCase();
        if (dropTags.has(tag)) {
          node.remove();
          return;
        }
        if (!allowedTags.has(tag)) {
          const parent = node.parentNode;
          if (!parent) return;
          while (node.firstChild) {
            parent.insertBefore(node.firstChild, node);
          }
          parent.removeChild(node);
          return;
        }

        Array.from(node.attributes).forEach(attr => {
          const name = attr.name.toLowerCase();
          if (tag === "a" && ["href", "title"].includes(name)) return;
          node.removeAttribute(attr.name);
        });

        if (tag === "a") {
          const href = node.getAttribute("href");
          if (!this.isSafeLink(href)) {
            node.removeAttribute("href");
          } else {
            node.setAttribute("target", "_blank");
            node.setAttribute("rel", "noopener noreferrer");
          }
        }
      });

      return template.innerHTML;
    },

    renderMarkdown(content, { narrator = false } = {}) {
      const source = narrator ? this.normalizeNarratorText(content) : String(content || "");
      return this.sanitizeHtml(marked.parse(source));
    },

    extractNarratorField(content, fields) {
      const wanted = new Set(fields);
      for (const rawLine of String(content || "").split("\n")) {
        const { key, value } = this.parseNarratorLine(rawLine);
        if (!wanted.has(key)) continue;
        if (value) return value;
      }
      return "";
    },

    messageGroups() {
      const groups = [];
      let current = null;

      this.messages.forEach(message => {
        if (message.kind === "narrator") {
          const payload = message.payload || null;
          current = {
            key: `scene-${message.id}`,
            title: (payload && payload.location) || this.extractNarratorField(message.content, ["地点", "场景"]) || "新的场面",
            time: payload ? [payload.date, payload.time].filter(Boolean).join(" ") : this.extractNarratorField(message.content, ["时间"]),
            narrator: message,
            messages: [message],
          };
          groups.push(current);
          return;
        }

        if (!current) {
          current = {
            key: `open-${message.id}`,
            title: "",
            time: "",
            narrator: null,
            messages: [],
          };
          groups.push(current);
        }
        current.messages.push(message);
      });

      return groups;
    },

    normalizeNarratorPayload(payload) {
      if (!payload || typeof payload !== "object") return null;
      const present = payload.present_characters && typeof payload.present_characters === "object"
        ? payload.present_characters
        : {};
      return {
        targets: Array.isArray(payload.targets) ? payload.targets : [],
        date: String(payload.date || "").trim(),
        time: String(payload.time || "").trim(),
        location: String(payload.location || "").trim(),
        present_characters: Object.fromEntries(
          Object.entries(present)
            .map(([name, value]) => [String(name || "").trim(), String(value || "").trim()])
            .filter(([name, value]) => name && value)
        ),
        scene_description: String(payload.scene_description || "").trim(),
        new_characters: Array.isArray(payload.new_characters) ? payload.new_characters : [],
      };
    },

    presenceEntries(payload) {
      const present = payload && payload.present_characters && typeof payload.present_characters === "object"
        ? payload.present_characters
        : {};
      return Object.entries(present);
    },

    getAgentStyle(author) {
      const key = author || "角色";
      if (!this.agentStyleByAuthor[key]) {
        const index = Object.keys(this.agentStyleByAuthor).length % this.agentPalette.length;
        this.agentStyleByAuthor[key] = this.agentPalette[index];
      }
      return this.agentStyleByAuthor[key];
    },

    _makeMessage(kind, author, content, turn = 0, { payload = null } = {}) {
      const normalizedPayload = kind === "narrator" ? this.normalizeNarratorPayload(payload) : null;
      const text = String(content || "").trim();
      if (!text && !normalizedPayload) return null;
      const resolvedAuthor = kind === "player" ? "你" : (author || "旁白");
      const style = kind === "agent" ? this.getAgentStyle(resolvedAuthor) : {
        color: "#b45a64",
        border: "rgba(180, 90, 100, 0.18)",
        background: kind === "player" ? "rgba(180, 90, 100, 0.08)" : "rgba(255, 251, 247, 0.88)",
      };
      return {
        id: this.nextMessageId++,
        kind,
        author: resolvedAuthor,
        color: style.color,
        border: style.border,
        background: style.background,
        content: text,
        payload: normalizedPayload,
        html: normalizedPayload ? "" : this.renderMarkdown(text, { narrator: kind === "narrator" }),
        turn: Number(turn) || 0,
      };
    },

    _buildHistoryMessages(items) {
      const built = [];
      let minTurn = null;
      (items || []).forEach(message => {
        const role = message.role || "";
        let kind = "agent";
        if (role === "player") kind = "player";
        else if (role === "narrator") kind = "narrator";
        const author = kind === "player" ? "你" : (message.author || role);
        const turn = Number(message.turn) || 0;
        const obj = this._makeMessage(kind, author, message.content, turn, { payload: message.payload || null });
        if (!obj) return;
        built.push(obj);
        if (turn > 0 && (minTurn === null || turn < minTurn)) minTurn = turn;
      });
      return { built, minTurn };
    },

    isNearLatest(buffer = 120) {
      const el = this.$refs.messages;
      if (!el) return true;
      return el.scrollHeight - el.scrollTop - el.clientHeight <= buffer;
    },

    updateLatestState() {
      this.nearLatest = this.isNearLatest();
      if (this.nearLatest) {
        this.unreadNewMessages = false;
      }
    },

    shouldAutoFollowLatest() {
      return !this.$refs.messages || this.nearLatest;
    },

    shouldShowJumpLatest() {
      return this.messages.length > 0 && !this.isInputMode() && (!this.nearLatest || this.unreadNewMessages);
    },

    jumpToLatest() {
      this.scrollToLatest({ settle: true });
    },

    _afterPush(shouldFollow) {
      if (shouldFollow) {
        this.scrollToLatest();
      } else {
        this.unreadNewMessages = true;
      }
    },

    addMessage(kind, author, content, turn = 0, { forceScroll = false, payload = null } = {}) {
      const message = this._makeMessage(kind, author, content, turn, { payload });
      if (!message) return;
      const shouldFollow = forceScroll || this.shouldAutoFollowLatest();
      this.messages.push(message);
      this._afterPush(shouldFollow);
    },

    addSystemMessage({ title = "系统", name = "", identity = "", characterId = "" } = {}) {
      const displayName = String(name || characterId || "新角色").trim();
      const identityText = String(identity || "").trim();
      const shouldFollow = this.shouldAutoFollowLatest();

      this.messages.push({
        id: this.nextMessageId++,
        kind: "system",
        title: String(title || "系统").trim(),
        name: displayName,
        identity: identityText,
        characterId: String(characterId || "").trim(),
      });

      this._afterPush(shouldFollow);
    },

    addHistory(messages) {
      const { built, minTurn } = this._buildHistoryMessages(messages);
      if (!built.length) return;
      this.messages.push(...built);
      if (minTurn !== null) this.oldestLoadedTurn = minTurn;
      this.scrollToLatest();
    },

    scrollToLatest({ settle = false } = {}) {
      const stick = () => {
        const el = this.$refs.messages;
        if (el) el.scrollTop = el.scrollHeight;
      };
      this.$nextTick(() => {
        requestAnimationFrame(() => {
          requestAnimationFrame(stick);
        });
      });
      setTimeout(stick, 250);
      if (settle) {
        [80, 500, 1000].forEach(delay => setTimeout(stick, delay));
      }
      this.unreadNewMessages = false;
      this.nearLatest = true;
    },

    prependHistory(messages) {
      const { built, minTurn } = this._buildHistoryMessages(messages);
      if (!built.length) return [];
      this.messages.unshift(...built);
      if (minTurn !== null && (this.oldestLoadedTurn == null || minTurn < this.oldestLoadedTurn)) {
        this.oldestLoadedTurn = minTurn;
      }
      return built;
    },

    async loadOlderHistory() {
      if (this.loadingOlder || this.historyExhausted) return;
      if (this.oldestLoadedTurn == null || this.oldestLoadedTurn <= 1) {
        this.historyExhausted = true;
        return;
      }
      this.loadingOlder = true;
      const container = this.$refs.messages;
      const prevHeight = container ? container.scrollHeight : 0;
      const prevTop = container ? container.scrollTop : 0;
      try {
        const requestLimit = 30;
        const response = await this.fetchJson(`/api/history?before_turn=${this.oldestLoadedTurn}&limit=${requestLimit}`);
        const fetched = response.messages || [];
        if (!fetched.length) {
          this.historyExhausted = true;
          return;
        }
        const inserted = this.prependHistory(fetched);
        if (fetched.length < requestLimit) {
          this.historyExhausted = true;
        }
        if (!inserted.length) return;
        await this.$nextTick();
        if (container) {
          container.scrollTop = container.scrollHeight - prevHeight + prevTop;
          this.updateLatestState();
        }
      } catch (error) {
        this.setNotice("加载更早的对话失败。", "error");
      } finally {
        this.loadingOlder = false;
      }
    },

    handleMessagesScroll() {
      if (!this.$refs.messages) return;
      const near = this.isNearLatest();
      if (near !== this.nearLatest) this.nearLatest = near;
      if (near) this.unreadNewMessages = false;
      if (this.$refs.messages.scrollTop < 80) {
        this.loadOlderHistory();
      }
    },

    async ensureGameDates() {
      if (this.gameDatesLoaded || this.gameDatesLoading) return;
      this.gameDatesLoading = true;
      try {
        const response = await this.fetchJson("/api/history/dates");
        this.gameDates = response.anchors || [];
        this.gameDatesLoaded = true;
      } catch (error) {
        this.setNotice("加载历史日期失败。", "error");
      } finally {
        this.gameDatesLoading = false;
      }
    },

    async toggleDateMenu() {
      this.dateMenuOpen = !this.dateMenuOpen;
      if (this.dateMenuOpen) {
        this.closeHistorySearch();
        await this.ensureGameDates();
        await this.$nextTick();
        this.positionDateMenu();
      }
    },

    positionDateMenu() {
      const btn = this.$refs.dateBtn;
      const menu = this.$refs.dateMenu;
      if (!btn || !menu) return;
      const rect = btn.getBoundingClientRect();
      menu.style.top = `${rect.bottom + 8}px`;
      menu.style.right = `${Math.max(8, window.innerWidth - rect.right)}px`;
    },

    async jumpToTurn(targetTurn) {
      const target = Number(targetTurn);
      if (!target || target < 1) return;
      this.dateMenuOpen = false;
      this.historySearchOpen = false;
      this.closeMobileMenu();
      const container = this.$refs.messages;
      let exists = this.messages.some(m => m.turn === target);
      let safety = 20;
      while (!exists && this.oldestLoadedTurn != null && this.oldestLoadedTurn > target && safety-- > 0) {
        const requestLimit = Math.min(200, Math.max(30, this.oldestLoadedTurn - target));
        try {
          const response = await this.fetchJson(`/api/history?before_turn=${this.oldestLoadedTurn}&limit=${requestLimit}`);
          const fetched = response.messages || [];
          if (!fetched.length) {
            this.historyExhausted = true;
            break;
          }
          this.prependHistory(fetched);
          if (fetched.length < requestLimit) this.historyExhausted = true;
        } catch (error) {
          this.setNotice("跳转加载失败。", "error");
          return;
        }
        exists = this.messages.some(m => m.turn === target);
      }
      if (!exists) return;
      await this.$nextTick();
      if (container) {
        const node = container.querySelector(`[data-turn="${target}"]`);
        if (node) {
          node.scrollIntoView({ behavior: "smooth", block: "start" });
          setTimeout(() => this.updateLatestState(), 260);
        }
      }
    },

    toggleHistorySearch() {
      if (this.historySearchOpen) {
        this.closeHistorySearch();
      } else {
        this.openHistorySearch();
      }
    },

    positionHistorySearch() {
      const btn = this.$refs.searchBtn;
      const popover = this.$refs.searchPopover;
      if (!btn || !popover) return;
      const rect = btn.getBoundingClientRect();
      const width = Math.min(430, window.innerWidth - 24);
      const left = Math.min(
        Math.max(12, rect.right - width),
        window.innerWidth - width - 12
      );
      popover.style.top = `${rect.bottom + 8}px`;
      popover.style.left = `${left}px`;
      popover.style.right = "auto";
      popover.style.width = `${width}px`;
    },

    openHistorySearch() {
      this.historySearchOpen = true;
      this.dateMenuOpen = false;
      this.$nextTick(() => {
        this.positionHistorySearch();
        this.$refs.historySearchInput?.focus();
        if (this.searchQuery.trim() && !this.searchResults.length && !this.searchLoading) {
          this.onSearchInput();
        }
      });
    },

    closeHistorySearch() {
      this.historySearchOpen = false;
    },

    clearHistorySearch({ keepOpen = false } = {}) {
      this.searchQuery = "";
      this.searchResults = [];
      this.searchLoading = false;
      this.historySearchOpen = keepOpen;
      if (this.searchTimer) {
        clearTimeout(this.searchTimer);
        this.searchTimer = null;
      }
      if (keepOpen) {
        this.$nextTick(() => this.$refs.historySearchInput?.focus());
      }
    },

    onSearchInput() {
      if (this.searchTimer) clearTimeout(this.searchTimer);
      const q = this.searchQuery.trim();
      if (!q) {
        this.searchResults = [];
        this.searchLoading = false;
        this.historySearchOpen = false;
        return;
      }
      if (!this.historySearchOpen) this.historySearchOpen = true;
      this.searchLoading = true;
      this.searchTimer = setTimeout(() => this.runSearch(q), 250);
    },

    async runSearch(q) {
      try {
        const response = await this.fetchJson(`/api/history/search?q=${encodeURIComponent(q)}&limit=50`);
        if (this.searchQuery.trim() !== q) return;
        this.searchResults = response.messages || [];
      } catch (error) {
        this.setNotice("搜索失败。", "error");
      } finally {
        if (this.searchQuery.trim() === q) {
          this.searchLoading = false;
        }
      }
    },

    buildSearchSnippet(content, query) {
      const text = String(content || "");
      const q = String(query || "");
      if (!q) return { before: text.slice(0, 60), match: "", after: "" };
      const idx = text.toLowerCase().indexOf(q.toLowerCase());
      if (idx === -1) return { before: text.slice(0, 60), match: "", after: "" };
      const start = Math.max(0, idx - 24);
      const end = Math.min(text.length, idx + q.length + 36);
      const beforeBody = text.slice(start, idx).replace(/\s+/g, " ");
      const matchBody = text.slice(idx, idx + q.length);
      const afterBody = text.slice(idx + q.length, end).replace(/\s+/g, " ");
      return {
        before: (start > 0 ? "…" : "") + beforeBody,
        match: matchBody,
        after: afterBody + (end < text.length ? "…" : ""),
      };
    },

    renderSearchSnippet(content, query) {
      const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      const { before, match, after } = this.buildSearchSnippet(content, query);
      return `${esc(before)}<mark>${esc(match)}</mark>${esc(after)}`;
    },

    resetConversation() {
      this.messages = [];
      this.choices = [];
      this.choiceStatus = "hidden";
      this.nextMessageId = 1;
      this.agentStyleByAuthor = {};
      this.observeMode = false;
      this.oldestLoadedTurn = null;
      this.historyExhausted = false;
      this.loadingOlder = false;
      this.gameDates = [];
      this.gameDatesLoaded = false;
      this.dateMenuOpen = false;
      this.searchQuery = "";
      this.searchResults = [];
      this.searchLoading = false;
      this.historySearchOpen = false;
      this.nearLatest = true;
      this.unreadNewMessages = false;
      this.characters = [];
      clearTimeout(this.searchTimer);
      this.searchTimer = null;
      this.clearNotice();
      this.resetComposer();
    },

    async startNewGame(storyId) {
      if (this.busy) return;
      if (this.consolidating) {
        this.setNotice("记忆整理进行中，请稍后再开始新故事。", "error");
        return;
      }
      this.cancelActiveStream({ silent: true });
      this.storyModalOpen = false;
      this.resetConversation();
      this.busy = true;

      try {
        const response = await this.postJson("/api/new_game", { story_id: storyId });
        if (response.intro) this.addMessage("narrator", "旁白", response.intro);
        if (response.opening) this.addMessage("narrator", "旁白", response.opening);
        this.choices = response.choices || [];
        this.choiceStatus = this.choices.length ? "ready" : "empty";
        this.characterCount = response.character_count || 0;
        this.hasSave = true;
        this.initialRecent = [];
        this.initialChoices = this.choices;
        await this.refreshSaves({ quiet: true });
        await this.fetchCharacters();
        this.pushToast("新故事已开始", "success");
      } catch (error) {
        this.storyModalOpen = true;
        this.setNotice("新故事启动失败，请稍后再试。", "error", true);
      } finally {
        this.busy = false;
      }
    },

    continueGame(recent, lastChoices, { silent = false } = {}) {
      this.storyModalOpen = false;
      this.resetConversation();
      this.choices = lastChoices || [];
      this.choiceStatus = this.choices.length ? "ready" : "hidden";
      this.addHistory(recent);
      this.scrollToLatest({ settle: true });
      if (this.isCompact) this.closeDrawer();
      if (!silent) {
        this.setNotice("已恢复最近进度。", "success");
      }
    },

    submitCurrentInput() {
      this.submitMessage(this.inputText);
    },

    handleInputKeydown(event) {
      if (this.isCompact) return;
      if (event.isComposing) return;
      if (event.key !== "Enter" || event.shiftKey) return;
      event.preventDefault();
      this.submitCurrentInput();
    },

    cancelActiveStream({ silent = false } = {}) {
      if (!this.activeStreamController) return;
      this.activeStreamController.abort();
      this.activeStreamController = null;
      this.activeStreamId += 1;
      this.busy = false;
      if (!silent) {
        this.pushToast("当前请求已取消。", "success");
      }
    },

    async submitMessage(text) {
      const message = String(text || "").trim();
      if (!message || this.busy) return;
      if (this.activeStreamController) {
        this.activeStreamController.abort();
        this.activeStreamController = null;
      }

      if (this.isCompact) {
        this.blurComposer();
      }
      if (this.observeMode) {
        this.addMessage("narrator", "旁白", `*（观察模式：${message}）*`, 0, { forceScroll: true });
      } else {
        this.addMessage("player", "你", message, 0, { forceScroll: true });
      }
      this.resetComposer();
      this.choices = [];
      this.choiceStatus = this.observeMode ? "hidden" : "loading";
      this.busy = true;
      const streamId = this.activeStreamId + 1;
      this.activeStreamId = streamId;

      try {
        await this.streamChat(message, streamId);
      } catch (error) {
        if (this.activeStreamId === streamId && error.name !== "AbortError") {
          this.setNotice("消息发送失败，请检查连接后重试。", "error", true);
          this.pushToast("发送失败", "error");
        }
      } finally {
        if (this.activeStreamId === streamId) {
          if (this.choiceStatus === "loading") {
            this.choiceStatus = "empty";
          }
          this.busy = false;
          this.activeStreamController = null;
        }
      }
    },

    async streamChat(message, streamId) {
      const controller = new AbortController();
      this.activeStreamController = controller;
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, mode: this.observeMode ? "observe" : "participate" }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`/api/chat failed: ${response.status}`);
      }
      if (this.activeStreamId !== streamId) {
        return;
      }
      if (!response.body) {
        throw new Error("SSE body is empty");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (this.activeStreamId !== streamId) {
          await reader.cancel();
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() || "";
        chunks.forEach(chunk => this.handleStreamPart(chunk, streamId));
      }

      if (buffer.trim()) {
        this.handleStreamPart(buffer, streamId);
      }
    },

    handleStreamPart(part, streamId = this.activeStreamId) {
      if (streamId !== this.activeStreamId) return;
      const payload = part
        .split("\n")
        .filter(line => line.startsWith("data:"))
        .map(line => line.replace(/^data:\s*/, ""))
        .join("");

      if (!payload) return;

      try {
        const event = JSON.parse(payload);
        if (event.type === "narrator") {
          this.addMessage("narrator", event.author || "旁白", event.content, 0, { payload: event.payload || null });
        } else if (event.type === "system") {
          this.addSystemMessage({
            title: event.title || "系统",
            name: event.name || "",
            identity: event.identity || "",
            characterId: event.character_id || "",
          });
          if (event.title === "角色已创建") {
            this.fetchCharacters();
          }
        } else if (event.type === "agent") {
          this.addMessage("agent", event.author, event.content);
        } else if (event.type === "choices") {
          this.choices = event.choices || [];
          this.choiceStatus = this.choices.length ? "ready" : "empty";
          if (this.choices.length) {
            this.setNotice("这一轮已生成新的建议行动。", "success");
          } else {
            this.setNotice("这一轮没有预设选项，你可以直接输入下一句。", "success");
          }
        } else if (event.type === "done") {
          this.setConsolidating(Boolean(event.consolidating));
          this.fetchCharacters();
        } else if (event.type === "response_done") {
          this.busy = false;
          this.setConsolidating(Boolean(event.consolidating));
        }
      } catch (error) {
        this.setNotice("实时消息解析失败，界面可能没有完整更新。", "error", true);
      }
    },

    setConsolidating(running) {
      this.consolidating = running;
      if (running) {
        this.startConsolidationPolling();
      } else {
        this.stopConsolidationPolling();
      }
    },

    startConsolidationPolling() {
      if (this.consolidatingPollTimer) return;
      // 用 generation 序号防止 stop 后正在飞的 tick 在 await 完成后又把自己排进下一轮
      const gen = (this.consolidatingPollGen || 0) + 1;
      this.consolidatingPollGen = gen;
      const tick = async () => {
        if (this.consolidatingPollGen !== gen) return;
        try {
          const response = await fetch("/api/status");
          if (this.consolidatingPollGen !== gen) return;
          if (response.ok) {
            const data = await response.json();
            if (!data.consolidating) {
              this.stopConsolidationPolling();
              this.consolidating = false;
              if (data.new_episodes && data.new_episodes.length) {
                for (const ep of data.new_episodes) {
                  const title = ep.title ? `「${ep.title}」` : "";
                  this.pushToast(`${ep.display_name}记忆新增事件${title}`, "success");
                }
              }
              return;
            }
          }
        } catch (error) {
          // 网络抖动忽略，下次再试
          if (this.consolidatingPollGen !== gen) return;
        }
        this.consolidatingPollTimer = setTimeout(tick, 1500);
      };
      this.consolidatingPollTimer = setTimeout(tick, 1500);
    },

    stopConsolidationPolling() {
      this.consolidatingPollGen = (this.consolidatingPollGen || 0) + 1;
      if (this.consolidatingPollTimer) {
        clearTimeout(this.consolidatingPollTimer);
        this.consolidatingPollTimer = null;
      }
    },

    normalizeSave(save) {
      return typeof save === "string"
        ? { filename: save, display: save, focus: "" }
        : {
            ...save,
            filename: save.filename,
            display: save.display_time || save.filename,
            focus: save.focus || "",
          };
    },

    storyLabel(storyId) {
      const story = this.stories.find(item => item.id === storyId);
      return story?.label || story?.title || storyId || "未知世界观";
    },

    storySummary(storyId) {
      const story = this.stories.find(item => item.id === storyId);
      return story?.summary || "这个世界观下的存档会按父子关系显示为分支树。";
    },

    normalizeWorld(world) {
      const storyId = world.story_id || "unknown";
      return {
        ...world,
        story_id: storyId,
        display_name: this.storyLabel(storyId),
        roots: world.roots || [],
        orphans: world.orphans || [],
        save_count: Number(world.save_count) || 0,
        root_count: Number(world.root_count) || 0,
        orphan_count: Number(world.orphan_count) || 0,
      };
    },

    worldTabs() {
      const byId = new Map();
      this.stories.forEach(story => {
        byId.set(story.id, {
          story_id: story.id,
          display_name: story.label || story.title || story.id,
          roots: [],
          orphans: [],
          save_count: 0,
          root_count: 0,
          orphan_count: 0,
        });
      });
      this.worlds.forEach(world => {
        byId.set(world.story_id, {
          ...(byId.get(world.story_id) || {}),
          ...world,
          display_name: this.storyLabel(world.story_id),
        });
      });
      return Array.from(byId.values());
    },

    ensureActiveWorld() {
      const tabs = this.worldTabs();
      if (!tabs.length) {
        this.activeWorldId = "";
        return;
      }
      if (this.activeWorldId && tabs.some(world => world.story_id === this.activeWorldId)) return;
      if (this.currentStoryId && tabs.some(world => world.story_id === this.currentStoryId)) {
        this.activeWorldId = this.currentStoryId;
        return;
      }
      this.activeWorldId = tabs[0].story_id;
    },

    activeWorld() {
      const tabs = this.worldTabs();
      return tabs.find(world => world.story_id === this.activeWorldId) || tabs[0] || null;
    },

    activeWorldLabel() {
      const world = this.activeWorld();
      return world?.display_name || "世界观";
    },

    activeWorldDescription() {
      const world = this.activeWorld();
      return this.storySummary(world?.story_id || "");
    },

    worldlineGameLatestKey(node) {
      if (!node) return "";
      const childLatest = (node.children || []).reduce((latest, child) => {
        const childKey = this.worldlineGameLatestKey(child);
        return childKey > latest ? childKey : latest;
      }, "");
      const ownKey = node.latest_created_at || node.created_at || node.display || node.filename || "";
      return ownKey > childLatest ? ownKey : childLatest;
    },

    worldlineGameRoots() {
      const world = this.activeWorld();
      if (!world) return [];
      const games = [
        ...(world.roots || []).map(node => ({
          node,
          orphan: false,
        })),
        ...(world.orphans || []).map(node => ({
          node,
          orphan: true,
        })),
      ];
      games.sort((a, b) => this.worldlineGameLatestKey(b.node).localeCompare(this.worldlineGameLatestKey(a.node)));
      let gameIndex = 0;
      let orphanIndex = 0;
      return games.map((game, index) => {
        if (game.orphan) {
          orphanIndex += 1;
          return { ...game, index, label: `断裂分支 ${orphanIndex}` };
        }
        gameIndex += 1;
        return { ...game, index, label: `Game ${gameIndex}` };
      });
    },

    layoutWorldlineTree(root, { orphan = false, index = 0, label = "Game" } = {}) {
      const cardWidth = 196;
      const cardHeight = 126;
      const cardOffsetX = 26;
      const cardOffsetY = -38;
      const siblingGap = 276;
      const levelGap = 198;
      const padX = 48;
      const padY = 62;
      let leafIndex = 0;
      let maxDepth = 0;

      const build = (node, depth = 0) => {
        maxDepth = Math.max(maxDepth, depth);
        const childLayouts = (node.children || []).map(child => build(child, depth + 1));
        const x = childLayouts.length
          ? childLayouts.reduce((sum, child) => sum + child.x, 0) / childLayouts.length
          : padX + leafIndex++ * siblingGap;
        return {
          node,
          depth,
          x,
          y: padY + depth * levelGap,
          children: childLayouts,
        };
      };

      const rootLayout = build(root);
      const nodes = [];
      const links = [];
      let order = 0;
      let maxRight = 0;
      let maxBottom = 0;

      const flatten = (layout, parent = null) => {
        const dotX = layout.x;
        const dotY = layout.y;
        const cardX = dotX + cardOffsetX;
        const cardY = Math.max(16, dotY + cardOffsetY);
        const item = {
          ...layout.node,
          depth: layout.depth,
          dotX,
          dotY,
          cardX,
          cardY,
          order: order++,
          child_count: (layout.node.children || []).length,
          can_delete: Boolean(parent) && !(layout.node.children || []).length,
          parent_missing: Boolean(layout.node.parent_missing || (orphan && layout.depth === 0)),
        };
        nodes.push(item);
        maxRight = Math.max(maxRight, cardX + cardWidth, dotX + 18);
        maxBottom = Math.max(maxBottom, cardY + cardHeight, dotY + 18);

        if (parent) {
          const startX = parent.x;
          const startY = parent.y;
          const endX = dotX;
          const endY = dotY;
          const midY = startY + Math.max(42, (endY - startY) * 0.5);
          const linkId = `${parent.node.save_id || parent.node.filename}->${layout.node.save_id || layout.node.filename}`;
          links.push({
            id: linkId,
            path: `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`,
            startX,
            startY,
            endX,
            endY,
          });
        }

        layout.children.forEach(child => flatten(child, layout));
      };

      flatten(rootLayout);

      const width = Math.max(330, maxRight + 44, padX * 2 + cardWidth + Math.max(0, leafIndex - 1) * siblingGap);
      const height = Math.max(250, maxBottom + 52, padY * 2 + cardHeight + maxDepth * levelGap);

      return {
        id: root.save_id || root.filename || `${label}-${index}`,
        index,
        root,
        title: root.title || root.focus || root.filename || label,
        kicker: orphan ? label : label,
        orphan,
        nodes,
        links,
        width,
        height,
        frameWidth: Math.max(320, width),
        frameHeight: Math.max(290, height + 56),
      };
    },

    activeWorldGames() {
      return this.worldlineGameRoots().map(item => this.layoutWorldlineTree(item.node, item));
    },

    worldlineLinkSvg(links) {
      return (links || []).map(link => {
        const path = String(link.path || "");
        const startX = Number(link.startX);
        const startY = Number(link.startY);
        const endX = Number(link.endX);
        const endY = Number(link.endY);
        if (!path || !Number.isFinite(startX) || !Number.isFinite(startY) || !Number.isFinite(endX) || !Number.isFinite(endY)) {
          return "";
        }
        return [
          `<path class="worldline-game-link-shadow" d="${path}"></path>`,
          `<path class="worldline-game-link" d="${path}"></path>`,
          `<circle class="worldline-game-link-joint" cx="${startX}" cy="${startY}" r="3.2"></circle>`,
          `<circle class="worldline-game-link-joint" cx="${endX}" cy="${endY}" r="3.2"></circle>`,
        ].join("");
      }).join("");
    },

    clampWorldlineZoom(value) {
      const scale = Number(value);
      if (!Number.isFinite(scale)) return 1;
      return Math.min(2.4, Math.max(0.45, scale));
    },

    worldlineZoomValue() {
      return this.clampWorldlineZoom(this.worldlineZoom || 1);
    },

    worldlineZoomLabel() {
      return `${Math.round(this.worldlineZoomValue() * 100)}%`;
    },

    worldlineCssNumber(name, fallback) {
      const shell = this.$refs?.worldlineShell;
      if (!shell) return fallback;
      const value = Number.parseFloat(getComputedStyle(shell).getPropertyValue(name));
      return Number.isFinite(value) ? value : fallback;
    },

    worldlineRowGeometry() {
      return {
        gap: this.worldlineCssNumber("--worldline-row-gap", 72),
        padX: this.worldlineCssNumber("--worldline-row-pad-x", 34),
        padY: this.worldlineCssNumber("--worldline-row-pad-y", 30),
      };
    },

    worldlineSurfaceMetrics() {
      const games = this.activeWorldGames();
      const { gap, padX, padY } = this.worldlineRowGeometry();
      const { contentWidth, contentHeight } = games.reduce(
        (acc, game) => ({
          contentWidth: acc.contentWidth + (Number(game.frameWidth) || 0),
          contentHeight: Math.max(acc.contentHeight, Number(game.frameHeight) || 0),
        }),
        { contentWidth: 0, contentHeight: 0 }
      );
      const width = padX * 2 + contentWidth + Math.max(0, games.length - 1) * gap;
      const height = padY * 2 + contentHeight + 12;
      return {
        width: Math.max(320, width),
        height: Math.max(260, height),
      };
    },

    worldlineSurfaceStyle() {
      const zoom = this.worldlineZoomValue();
      const metrics = this.worldlineSurfaceMetrics();
      const viewport = this.$refs?.worldlineScroll;
      const width = Math.max(viewport?.clientWidth || 0, metrics.width * zoom);
      const height = Math.max(viewport?.clientHeight || 0, metrics.height * zoom);
      return `width: ${Math.ceil(width)}px; height: ${Math.ceil(height)}px;`;
    },

    worldlineRowStyle() {
      return `transform: scale(${this.worldlineZoomValue()});`;
    },

    setWorldlineZoom(nextZoom, anchor = null) {
      const target = this.$refs?.worldlineScroll;
      const oldZoom = this.worldlineZoomValue();
      const next = this.clampWorldlineZoom(nextZoom);
      if (Math.abs(next - oldZoom) < 0.001) return;

      let anchorX = 0;
      let anchorY = 0;
      let logicalX = 0;
      let logicalY = 0;
      if (target) {
        const rect = target.getBoundingClientRect();
        anchorX = anchor?.x ?? rect.width / 2;
        anchorY = anchor?.y ?? rect.height / 2;
        logicalX = (target.scrollLeft + anchorX) / oldZoom;
        logicalY = (target.scrollTop + anchorY) / oldZoom;
      }

      this.worldlineZoom = next;

      if (target) {
        this.$nextTick(() => {
          target.scrollLeft = logicalX * next - anchorX;
          target.scrollTop = logicalY * next - anchorY;
        });
      }
    },

    zoomWorldlineBy(multiplier, anchor = null) {
      this.setWorldlineZoom(this.worldlineZoomValue() * multiplier, anchor);
    },

    resetWorldlineZoom() {
      this.setWorldlineZoom(1);
    },

    handleWorldlineWheel(event) {
      if (!this.worldlineOpen || this.saveLoading || this.saveError) return;
      event.preventDefault();

      const target = event.currentTarget;
      let deltaY = Number(event.deltaY) || 0;
      if (event.deltaMode === 1) deltaY *= 16;
      if (event.deltaMode === 2) deltaY *= target?.clientHeight || 600;
      if (!deltaY) return;

      const rect = target.getBoundingClientRect();
      const anchor = {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top,
      };
      this.zoomWorldlineBy(Math.exp(-deltaY * 0.0011), anchor);
    },

    startWorldlineDrag(event) {
      if (!this.worldlineOpen || event.button !== 0) return;
      if (event.target.closest("button, a, input, textarea, select")) return;
      const target = this.$refs?.worldlineScroll;
      if (!target) return;
      this.worldlineDrag = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        scrollLeft: target.scrollLeft,
        scrollTop: target.scrollTop,
      };
      target.classList.add("dragging");
      target.setPointerCapture?.(event.pointerId);
    },

    dragWorldline(event) {
      const drag = this.worldlineDrag;
      const target = this.$refs?.worldlineScroll;
      if (!drag || !target || drag.pointerId !== event.pointerId) return;
      event.preventDefault();
      target.scrollLeft = drag.scrollLeft - (event.clientX - drag.startX);
      target.scrollTop = drag.scrollTop - (event.clientY - drag.startY);
    },

    stopWorldlineDrag(event) {
      const drag = this.worldlineDrag;
      const target = this.$refs?.worldlineScroll;
      if (!drag || (event?.pointerId != null && drag.pointerId !== event.pointerId)) return;
      if (target?.hasPointerCapture?.(drag.pointerId)) {
        target.releasePointerCapture(drag.pointerId);
      }
      target?.classList.remove("dragging");
      this.worldlineDrag = null;
    },

    findWorldlineNode(saveId) {
      if (!saveId) return null;
      for (const world of this.worlds) {
        const stack = [...(world.roots || []), ...(world.orphans || [])];
        while (stack.length) {
          const node = stack.shift();
          if (node?.save_id === saveId) return node;
          stack.push(...(node?.children || []));
        }
      }
      return null;
    },

    currentWorldlineText() {
      if (!this.currentSaveId) return "当前进度尚未保存为节点";
      const node = this.findWorldlineNode(this.currentSaveId);
      if (!node) return this.currentSaveId;
      return node.title || node.focus || node.filename;
    },

    worldlineNodeKicker(item) {
      if (item.save_id && item.save_id === this.currentSaveId) return "当前节点";
      if (item.parent_missing) return "父节点缺失";
      if (!item.depth) return "根节点";
      if (item.child_count) return `${item.child_count} 条分支`;
      return "节点";
    },

    captureWorldlineGameRects() {
      if (!this.worldlineOpen) return {};
      const rects = {};
      document.querySelectorAll(".worldline-page .worldline-game[data-worldline-game-id]").forEach(element => {
        rects[element.dataset.worldlineGameId] = element.getBoundingClientRect();
      });
      return rects;
    },

    playWorldlineGameFlip(beforeRects) {
      if (!this.worldlineOpen || !beforeRects || !Object.keys(beforeRects).length) return;

      requestAnimationFrame(() => {
        const moved = [];
        document.querySelectorAll(".worldline-page .worldline-game[data-worldline-game-id]").forEach(element => {
          const before = beforeRects[element.dataset.worldlineGameId];
          if (!before) return;

          const after = element.getBoundingClientRect();
          const zoom = this.worldlineZoomValue();
          const dx = (before.left - after.left) / zoom;
          const dy = (before.top - after.top) / zoom;
          if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) return;

          element.style.transition = "none";
          element.style.transform = `translate(${dx}px, ${dy}px)`;
          element.style.opacity = "0.98";
          moved.push(element);
        });

        if (!moved.length) return;
        requestAnimationFrame(() => {
          moved.forEach(element => {
            element.style.transition = "transform 320ms cubic-bezier(0.16, 1, 0.3, 1), opacity 220ms ease";
            element.style.transform = "translate(0, 0)";
            element.style.opacity = "1";
            window.setTimeout(() => {
              element.style.transition = "";
              element.style.transform = "";
              element.style.opacity = "";
            }, 360);
          });
        });
      });
    },

    worldlineNodeLatestKey(node) {
      if (!node) return "";
      const childLatest = (node.children || []).reduce((latest, child) => {
        const childKey = this.worldlineNodeLatestKey(child);
        return childKey > latest ? childKey : latest;
      }, "");
      const ownKey = node.created_at || node.display || node.display_time || node.filename || "";
      return ownKey > childLatest ? ownKey : childLatest;
    },

    countWorldlineNodes(nodes) {
      return (nodes || []).reduce(
        (total, node) => total + 1 + this.countWorldlineNodes(node.children || []),
        0,
      );
    },

    pruneWorldlineNodes(nodes, deletedFilenames) {
      return (nodes || []).flatMap(node => {
        if (deletedFilenames.has(node.filename)) return [];
        const children = this.pruneWorldlineNodes(node.children || [], deletedFilenames);
        const next = {
          ...node,
          children,
          child_count: children.length,
        };
        next.latest_created_at = this.worldlineNodeLatestKey(next);
        return [next];
      });
    },

    removeSaveFilenamesFromWorlds(filenames) {
      const deletedFilenames = new Set((filenames || []).filter(Boolean));
      if (!deletedFilenames.size) return;

      let removedCurrentNode = false;
      const checkCurrent = nodes => {
        (nodes || []).forEach(node => {
          if (deletedFilenames.has(node.filename) && node.save_id && node.save_id === this.currentSaveId) {
            removedCurrentNode = true;
          }
          checkCurrent(node.children || []);
        });
      };
      this.worlds.forEach(world => {
        checkCurrent(world.roots || []);
        checkCurrent(world.orphans || []);
      });

      this.saves = this.saves.filter(save => !deletedFilenames.has(save.filename));
      this.worlds = this.worlds.map(world => {
        const roots = this.pruneWorldlineNodes(world.roots || [], deletedFilenames);
        const orphans = this.pruneWorldlineNodes(world.orphans || [], deletedFilenames);
        return {
          ...world,
          roots,
          orphans,
          save_count: this.countWorldlineNodes(roots) + this.countWorldlineNodes(orphans),
          root_count: roots.length,
          orphan_count: orphans.length,
        };
      });

      if (removedCurrentNode) {
        this.currentSaveId = "";
      }
      this.ensureActiveWorld();
    },

    syncSavesAfterWorldlineAnimation() {
      window.setTimeout(() => {
        this.refreshSaves({ quiet: true, silent: true });
      }, 380);
    },

    async fetchCharacters() {
      const seq = ++this._fetchCharactersSeq;
      // 已有数据时静默刷新，不触发 loading 闪烁；首次加载才显示骨架屏
      const isFirstLoad = this.characters.length === 0;
      if (isFirstLoad) {
        this.charactersLoading = true;
      }
      try {
        const response = await this.fetchJson("/api/characters");
        // 只采纳最新一次请求的结果，丢弃被后续请求超越的旧响应
        if (seq === this._fetchCharactersSeq) {
          this.characters = response.characters || [];
        }
      } catch (_error) {
        // 后台刷新失败时保留旧数据，不清空已展示内容
      } finally {
        this.charactersLoading = false;
      }
    },

    async refreshSaves({ quiet = false, silent = false } = {}) {
      if (!silent) {
        this.saveLoading = true;
      }
      this.saveError = "";
      try {
        const response = await this.fetchJson("/api/saves");
        this.saves = (response.saves || []).map(save => this.normalizeSave(save));
        this.worlds = (response.worlds || []).map(world => this.normalizeWorld(world));
        this.currentSaveId = response.current_save_id || "";
        this.currentStoryId = response.current_story_id || "";
        this.ensureActiveWorld();
      } catch (error) {
        this.saveError = "存档列表加载失败，请稍后再试。";
        if (!quiet) {
          this.setNotice(this.saveError, "error", true);
        }
      } finally {
        if (!silent) {
          this.saveLoading = false;
        }
      }
    },

    promptLoadSave(filename) {
      if (this.busy) {
        this.setNotice("请等待当前回合结束后再读档。", "error");
        return;
      }
      if (this.consolidating) {
        this.setNotice("记忆整理进行中，请稍后再读档。", "error");
        return;
      }
      this.openConfirm({
        title: "读取这个存档？",
        body: `当前运行时会恢复到 "${filename}" 记录的世界线节点。之后新建存档会从该节点继续分支。`,
        confirmText: "读取",
        onConfirm: async () => {
          await this.loadSave(filename);
        },
      });
    },

    promptDeleteGame(game) {
      if (this.busy) {
        this.setNotice("请等待当前回合结束后再删除 Game。", "error");
        return;
      }
      if (this.consolidating) {
        this.setNotice("记忆整理进行中，请稍后再删除 Game。", "error");
        return;
      }
      const rootFilename = game?.root?.filename || "";
      if (!rootFilename) return;
      this.openConfirm({
        title: "删除这个 Game？",
        body: `将永久删除 "${game.title || rootFilename}" 以及它下面的 ${game.nodes?.length || 1} 个存档节点。此操作无法撤销。`,
        confirmText: "删除",
        onConfirm: async () => {
          await this.deleteGame(rootFilename);
        },
      });
    },

    promptDeleteLeafSave(node) {
      if (this.busy) {
        this.setNotice("请等待当前回合结束后再删除存档。", "error");
        return;
      }
      if (this.consolidating) {
        this.setNotice("记忆整理进行中，请稍后再删除存档。", "error");
        return;
      }
      const filename = node?.filename || "";
      if (!filename) return;
      if ((node.child_count || 0) > 0) {
        this.setNotice("这个存档已经是其他分支的 parent，不能单独删除。", "error");
        return;
      }
      this.openConfirm({
        title: "删除这个存档节点？",
        body: `将永久删除 "${node.title || node.focus || filename}"。只有没有子分支的存档节点可以这样删除。`,
        confirmText: "删除",
        onConfirm: async () => {
          await this.deleteLeafSave(filename);
        },
      });
    },

    async deleteLeafSave(filename) {
      this.busy = true;
      try {
        const response = await this.deleteJson(`/api/save-node/${encodeURIComponent(filename)}`);
        const deleted = response.deleted?.length ? response.deleted : [filename];
        const beforeRects = this.captureWorldlineGameRects();
        this.removeSaveFilenamesFromWorlds(deleted);
        this.playWorldlineGameFlip(beforeRects);
        this.syncSavesAfterWorldlineAnimation();
        const count = response.deleted?.length || 0;
        this.pushToast(count ? "存档节点已删除" : "存档已删除", "success");
      } catch (error) {
        this.pushToast("删除失败：只能删除没有子分支的存档。", "error");
      } finally {
        this.busy = false;
      }
    },

    async deleteGame(rootFilename) {
      this.busy = true;
      try {
        const response = await this.deleteJson(`/api/save/${encodeURIComponent(rootFilename)}`);
        const deleted = response.deleted?.length ? response.deleted : [rootFilename];
        const beforeRects = this.captureWorldlineGameRects();
        this.removeSaveFilenamesFromWorlds(deleted);
        this.playWorldlineGameFlip(beforeRects);
        this.syncSavesAfterWorldlineAnimation();
        const count = response.deleted?.length || 0;
        this.pushToast(count ? `Game 已删除（${count} 个节点）` : "Game 已删除", "success");
      } catch (error) {
        this.pushToast("删除失败", "error");
      } finally {
        this.busy = false;
      }
    },

    async loadSave(filename) {
      this.cancelActiveStream({ silent: true });
      this.busy = true;

      try {
        const response = await this.postJson("/api/load", { filename });
        if (!response.ok) {
          throw new Error("load failed");
        }
        this.hasSave = true;
        this.characterCount = response.character_count || 0;
        this.initialRecent = response.recent || [];
        this.initialChoices = response.last_choices || [];
        this.continueGame(response.recent, response.last_choices);
        await this.refreshSaves({ quiet: true });
        await this.fetchCharacters();
        this.closeWorldline();
        this.pushToast("存档已加载", "success");
      } catch (error) {
        this.setNotice("存档读取失败，请稍后再试。", "error", true);
        this.pushToast("读档失败", "error");
      } finally {
        this.busy = false;
      }
    },

    async saveGame() {
      if (this.busy || this.isSaving) return;
      if (!this.messages.length) {
        this.setNotice("还没有可保存的对话内容。", "error");
        return;
      }

      this.isSaving = true;
      try {
        const response = await this.postJson("/api/save", {});

        if (response.ok) {
          this.hasSave = true;
          await this.refreshSaves({ quiet: true });
          if (response.duplicate) {
            this.setNotice("当前进度已存在同名存档，无需重复保存。", "info");
            this.pushToast("世界线节点已存在，无需重复保存", "info");
          } else {
            const savedName = response.filename || "新节点";
            this.setNotice(`已记录世界线节点：${savedName}`, "success");
            this.pushToast("世界线节点已创建", "success");
          }
        } else {
          throw new Error(response.detail || "存档失败");
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "存档失败，请稍后再试。";
        this.setNotice(`存档失败：${message}`, "error", true);
        this.pushToast("存档失败", "error");
      } finally {
        this.isSaving = false;
      }
    },
  }));
});
