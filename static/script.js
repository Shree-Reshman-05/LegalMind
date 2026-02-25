// ==================== MOBILE NAVIGATION ====================
const mobileMenuBtn = document.getElementById('mobileMenuBtn');
const mobileCloseBtn = document.getElementById('mobileCloseBtn');
const sidebar = document.getElementById('sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');

// Open sidebar on mobile
if (mobileMenuBtn) {
    mobileMenuBtn.addEventListener('click', () => {
        sidebar.classList.add('open');
        sidebarOverlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    });
}

// Close sidebar functions
function closeMobileSidebar() {
    if (window.innerWidth <= 768) {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }
}

if (mobileCloseBtn) {
    mobileCloseBtn.addEventListener('click', closeMobileSidebar);
}

if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeMobileSidebar);
}

// Swipe to close sidebar on mobile
let touchStartX = 0;
let touchEndX = 0;

if (sidebar) {
    sidebar.addEventListener('touchstart', e => {
        touchStartX = e.changedTouches[0].screenX;
    });

    sidebar.addEventListener('touchend', e => {
        touchEndX = e.changedTouches[0].screenX;
        if (touchStartX - touchEndX > 50) { // Swipe left
            closeMobileSidebar();
        }
    });
}

// Close sidebar on window resize
window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
        sidebar.classList.remove('open');
        sidebarOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }
});

// ==================== BROWSER BACK BUTTON ====================
(function handleBackButton() {
    if (!history.state || history.state.page !== "chatbot") {
        history.pushState({ page: "chatbot" }, "", location.href);
    }

    window.addEventListener("popstate", function () {
        window.location.href = "/";
    });
})();

// ==================== GLOBAL STATE ====================
let isProcessing = false;
let chatSessions = [];
let currentSessionId = null;

// ==================== AVATAR HELPER ====================
function getAvatarHTML(sender) {
    if (sender === 'user') {
        return `
            <div class="avatar user-avatar">
                👤
            </div>
        `;
    }

    return `
        <div class="avatar bot-avatar">
            <img src="/static/law.png" alt="LegalMind">
        </div>
    `;
}

// ==================== CHAT SESSION MANAGEMENT ====================
function initializeChatHistory() {
    const savedSessions = localStorage.getItem('legalMindChatSessions');
    if (savedSessions) {
        chatSessions = JSON.parse(savedSessions);
    }
    
    if (chatSessions.length === 0) {
        createNewSession();
    } else {
        currentSessionId = chatSessions[0].id;
        loadSession(currentSessionId);
    }
    
    renderChatHistory();
}

function createNewSession() {
    const sessionId = Date.now();
    const newSession = {
        id: sessionId,
        title: 'New Conversation',
        timestamp: new Date().toISOString(),
        messages: []
    };
    
    chatSessions.unshift(newSession);
    currentSessionId = sessionId;
    saveSessions();
    
    // Clear conversation on backend for this new session
    clearBackendConversation(sessionId);
    
    return sessionId;
}

function saveSessions() {
    localStorage.setItem('legalMindChatSessions', JSON.stringify(chatSessions));
}

function getCurrentSession() {
    return chatSessions.find(session => session.id === currentSessionId);
}

function saveMessageToSession(html, sender, sources = null) {
    const session = getCurrentSession();
    if (session) {
        session.messages.push({
            html,
            sender,
            sources,
            timestamp: new Date().toISOString()
        });
        
        // Update session title based on first user message
        if (sender === 'user' && session.messages.filter(m => m.sender === 'user').length === 1) {
            const maxLength = window.innerWidth <= 768 ? 40 : 50;
            session.title = html.substring(0, maxLength) + (html.length > maxLength ? '...' : '');
        }
        
        saveSessions();
        renderChatHistory();
    }
}

function loadSession(sessionId) {
    const session = chatSessions.find(s => s.id === sessionId);
    if (!session) return;
    
    currentSessionId = sessionId;
    const chatStream = document.getElementById('chatStream');
    chatStream.innerHTML = '';
    
    // Load all messages from the session
    session.messages.forEach(msg => {
        addMessageToUI(msg.html, msg.sender, msg.sources);
    });
    
    renderChatHistory();
    closeMobileSidebar(); // Close sidebar on mobile after loading session
}

function renderChatHistory() {
    const historyList = document.getElementById('chatHistory');
    historyList.innerHTML = '';
    
    chatSessions.forEach(session => {
        const li = document.createElement('li');
        li.textContent = session.title;
        li.className = session.id === currentSessionId ? 'active' : '';
        li.onclick = () => loadSession(session.id);
        
        // Add delete button
        if (chatSessions.length > 1) {
            const deleteBtn = document.createElement('span');
            deleteBtn.innerHTML = ' ×';
            deleteBtn.style.float = 'right';
            deleteBtn.style.cursor = 'pointer';
            deleteBtn.style.fontWeight = 'bold';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deleteSession(session.id);
            };
            li.appendChild(deleteBtn);
        }
        
        historyList.appendChild(li);
    });
}

function deleteSession(sessionId) {
    if (chatSessions.length <= 1) {
        alert('Cannot delete the last session');
        return;
    }
    
    chatSessions = chatSessions.filter(s => s.id !== sessionId);
    
    // Clear backend conversation for deleted session
    clearBackendConversation(sessionId);
    
    // If deleted session was active, switch to another
    if (currentSessionId === sessionId) {
        currentSessionId = chatSessions[0].id;
        loadSession(currentSessionId);
    }
    
    saveSessions();
    renderChatHistory();
}

// ==================== BACKEND CONVERSATION MANAGEMENT ====================
async function clearBackendConversation(sessionId) {
    try {
        await fetch('/clear-conversation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId.toString() })
        });
        console.log(`🧹 Cleared backend conversation for session ${sessionId}`);
    } catch (error) {
        console.error('Error clearing backend conversation:', error);
    }
}

// ==================== INITIALIZE ON PAGE LOAD ====================
document.addEventListener('DOMContentLoaded', function () {
    const userInput = document.getElementById('userInput');

    userInput.addEventListener('input', function () {
        autoResize(this);
    });

    userInput.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    userInput.focus();
    initializeChatHistory();
    checkBackendStatus();
});

// Auto resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    const maxHeight = window.innerWidth <= 768 ? 120 : 200;
    textarea.style.height = Math.min(textarea.scrollHeight, maxHeight) + 'px';
}

// Backend status check
async function checkBackendStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();

        if (data.documents === 0) {
            addMessage(
                '⚠️ <strong>No documents loaded.</strong><br><br>' +
                'Please add documents to the <code>data</code> folder and restart.',
                'bot'
            );
        }
    } catch {
        addMessage(
            '❌ <strong>Cannot connect to backend.</strong><br>' +
            'Run <code>python app.py</code>',
            'bot'
        );
    }
}

// Send message with session context
async function sendMessage() {
    if (isProcessing) return;

    const input = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const text = input.value.trim();
    if (!text) return;

    isProcessing = true;
    sendBtn.disabled = true;

    addMessage(text, 'user');

    input.value = '';
    input.style.height = 'auto';

    const loadingId = showTypingIndicator();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                message: text,
                session_id: currentSessionId.toString()
            })
        });

        const data = await response.json();
        removeTypingIndicator(loadingId);
        addMessage(data.reply, 'bot', data.sources);
    } catch (error) {
        removeTypingIndicator(loadingId);
        addMessage('❌ Backend error. Check server.', 'bot');
        console.error('Error:', error);
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        input.focus();
    }
}

// ==================== ADD MESSAGE ====================
function addMessage(html, sender, sources = null) {
    addMessageToUI(html, sender, sources);
    saveMessageToSession(html, sender, sources);
}

function addMessageToUI(html, sender, sources = null) {
    const chatStream = document.getElementById('chatStream');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    messageDiv.innerHTML = `
        ${sender === 'bot' ? getAvatarHTML('bot') : ''}
        <div class="content">${html}</div>
        ${sender === 'user' ? getAvatarHTML('user') : ''}
    `;

    if (sender === 'bot' && sources && sources.length > 0) {
        const contentDiv = messageDiv.querySelector('.content');
        contentDiv.innerHTML += createSourcesHTML(sources);
    }

    chatStream.appendChild(messageDiv);
    scrollToBottom();
}

// Sources UI
function createSourcesHTML(sources) {
    let html = '<div class="sources-container">';
    html += '<div class="sources-title">📚 Sources:</div>';

    sources.forEach(src => {
        const score = (src.score * 100).toFixed(1);
        html += `<div class="source-item">• ${src.source} (${score}%)</div>`;
    });

    html += '</div>';
    return html;
}

// ==================== TYPING INDICATOR ====================
function showTypingIndicator() {
    const chatStream = document.getElementById('chatStream');
    const id = `loading-${Date.now()}`;

    const div = document.createElement('div');
    div.className = 'message bot-message';
    div.id = id;
    div.innerHTML = `
        <div class="avatar bot-avatar">
            <img src="/static/law.png" alt="LegalMind">
        </div>
        <div class="content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;

    chatStream.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    const chatStream = document.getElementById('chatStream');
    chatStream.scrollTop = chatStream.scrollHeight;
}

// ==================== NEW CHAT ====================
function newChat() {
    createNewSession();
    const chatStream = document.getElementById('chatStream');
    chatStream.innerHTML = '';
    addMessage('<strong>New session started.</strong><br>Ask me anything about the legal documents!', 'bot');
    renderChatHistory();
    closeMobileSidebar(); // Close sidebar on mobile after creating new chat
}

// ==================== THEME HANDLING ====================
function setTheme(themeName) {
    document.body.classList.remove('theme-obsidian', 'theme-slate', 'theme-paper');
    document.body.classList.add(`theme-${themeName}`);
    localStorage.setItem('legalMindTheme', themeName);
    
    // Haptic feedback on mobile
    if (navigator.vibrate) {
        navigator.vibrate(10);
    }
}

(function loadTheme() {
    const saved = localStorage.getItem('legalMindTheme');
    if (saved) setTheme(saved);
})();

// ==================== PREVENT DOUBLE-TAP ZOOM (MOBILE) ====================
let lastTouchEnd = 0;
document.addEventListener('touchend', function (event) {
    const now = Date.now();
    if (now - lastTouchEnd <= 300) {
        event.preventDefault();
    }
    lastTouchEnd = now;
}, false);

// ==================== DETECT DEVICE TYPE ====================
function isMobile() {
    return window.innerWidth <= 768;
}

console.log(`📱 Device: ${isMobile() ? 'Mobile' : 'Desktop'}`);