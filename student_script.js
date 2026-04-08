// ═══════════════════════════════════════════════════
// LegalMind Student Mode — JavaScript
// ═══════════════════════════════════════════════════

// ── State ──
let currentView = 'dashboard';
let allTopics = [];
let allCases = [];
let quizQuestions = [];
let quizAnswers = {};
let currentQuizIdx = 0;
let quizSubmitted = false;
let challengeTimer = null;
let timeLeft = 300;

// ── Init ──
document.addEventListener('DOMContentLoaded', () => {
    loadTheme();
    initMobileNav();
    navigate('dashboard');
});

// ── Navigation ──
function navigate(view) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    const el = document.getElementById(`view-${view}`);
    if (el) el.classList.add('active');

    const btn = document.querySelector(`[data-view="${view}"]`);
    if (btn) btn.classList.add('active');

    currentView = view;
    closeMobileSidebar();

    switch (view) {
        case 'dashboard': loadDashboard(); break;
        case 'learn': loadLearn(); break;
        case 'cases': loadCases(); break;
        case 'quiz': resetQuiz(); break;
        case 'daily': loadDailyChallenge(); break;
        case 'progress': loadProgress(); break;
    }
}

// ── DASHBOARD ──
async function loadDashboard() {
    try {
        const [progressRes, topicsRes, casesRes] = await Promise.all([
            fetch('/api/progress'),
            fetch('/api/topics'),
            fetch('/api/cases')
        ]);
        const progress = await progressRes.json();
        const topicsData = await topicsRes.json();
        const casesData = await casesRes.json();

        allTopics = topicsData.topics;
        allCases = casesData.cases;

        // Stats
        document.getElementById('statTopics').textContent = `${progress.topics_completed}/${progress.total_topics}`;
        document.getElementById('statCases').textContent = `${progress.case_accuracy}%`;
        document.getElementById('statQuiz').textContent = `${progress.avg_quiz_score}%`;
        document.getElementById('statStreak').textContent = progress.streak;

        // Sidebar
        updateSidebar(progress);

        // Topic preview (first 4)
        const dashTopics = document.getElementById('dashTopics');
        dashTopics.innerHTML = '';
        allTopics.slice(0, 4).forEach(t => {
            dashTopics.appendChild(buildTopicCard(t));
        });

        // Badges
        const dashBadges = document.getElementById('dashBadges');
        dashBadges.innerHTML = '';
        if (progress.badges.length === 0) {
            dashBadges.innerHTML = '<span class="no-badges">Complete topics and quizzes to earn badges 🏅</span>';
        } else {
            progress.badges.forEach(b => {
                const pill = document.createElement('div');
                pill.className = 'badge-pill';
                pill.innerHTML = `<span class="badge-icon">${badgeIcon(b)}</span>${b}`;
                dashBadges.appendChild(pill);
            });
        }

        // Daily teaser
        const teaser = document.getElementById('dashDaily');
        const dayRes = await fetch('/api/daily-challenge');
        const dayData = await dayRes.json();
        if (dayData.already_done) {
            teaser.querySelector('.teaser-title').textContent = "Challenge Complete!";
            teaser.querySelector('.teaser-sub').textContent = "Come back tomorrow for a new challenge.";
            teaser.querySelector('.teaser-icon').textContent = '✅';
        }

    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

function updateSidebar(progress) {
    const level = progress.level;
    const pts = progress.points;
    document.getElementById('sidebarLevel').textContent = level;
    document.getElementById('sidebarPts').textContent = `${pts} pts`;
    document.getElementById('streakCount').textContent = progress.streak;
    document.getElementById('mobilePts').textContent = `${pts} pts`;

    const maxPts = level === 'Beginner' ? 200 : level === 'Intermediate' ? 500 : 1000;
    const pct = Math.min((pts / maxPts) * 100, 100);
    document.getElementById('xpFill').style.width = pct + '%';
}

// ── LEARN ──
async function loadLearn() {
    if (allTopics.length === 0) {
        const res = await fetch('/api/topics');
        const data = await res.json();
        allTopics = data.topics;
    }
    renderTopicList(allTopics);
}

function filterTopics(filter) {
    document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-filter="${filter}"]`).classList.add('active');

    const filtered = filter === 'all' ? allTopics : allTopics.filter(t => t.level === filter);
    renderTopicList(filtered);
}

function renderTopicList(topics) {
    const list = document.getElementById('topicList');
    list.innerHTML = '';
    topics.forEach(t => {
        const item = document.createElement('div');
        item.className = 'topic-list-item';
        item.onclick = () => openTopic(t.id);
        item.innerHTML = `
            <div class="tli-icon">${t.icon}</div>
            <div class="tli-body">
                <div class="tli-title">${t.title} ${t.completed ? '✅' : ''}</div>
                <div class="tli-meta">
                    <span class="level-badge ${t.level}">${t.level}</span>
                    <span class="tli-time">⏱ ${t.estimated_time}</span>
                    <span>${t.category}</span>
                </div>
            </div>
            <div class="tli-arrow">→</div>
        `;
        list.appendChild(item);
    });
}

async function openTopic(topicId) {
    const res = await fetch(`/api/topic/${topicId}`);
    const data = await res.json();
    const topic = data.topic;

    const content = document.getElementById('topicDetailContent');
    const isCompleted = allTopics.find(t => t.id === topicId)?.completed;

    let html = `
        <div class="topic-detail-header">
            <div class="td-icon">${topic.icon}</div>
            <div class="td-info">
                <div class="td-title">${topic.title}</div>
                <div class="td-meta">
                    <span class="level-badge ${topic.level}">${topic.level}</span>
                    <span style="color:var(--text-2);font-size:0.8rem">${topic.category}</span>
                    <span style="color:var(--text-2);font-size:0.8rem">⏱ ${topic.estimated_time}</span>
                </div>
            </div>
            <button class="td-complete-btn ${isCompleted ? 'done' : ''}" id="completeBtn" onclick="completeTopic('${topic.id}')">
                ${isCompleted ? '✅ Completed' : '✓ Mark Complete'}
            </button>
        </div>
    `;

    topic.sections.forEach(sec => {
        html += buildSection(sec);
    });

    content.innerHTML = html;

    // show section
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-topic-detail').classList.add('active');
    document.getElementById('mainContent').scrollTop = 0;
}

function buildSection(sec) {
    const icons = { definition: '📖', elements: '🧩', example: '💡', case_reference: '📚', flowchart: '🔀', law_section: '§' };
    const icon = icons[sec.type] || '•';

    let inner = '';
    if (sec.type === 'definition') {
        inner = `<p class="block-text">${sec.content}</p>`;
    } else if (sec.type === 'elements') {
        inner = `<ul class="elements-list">${sec.items.map(i => `
            <li class="element-item">
                <div class="element-label">${i.label}</div>
                <div class="element-detail">${i.detail}</div>
            </li>`).join('')}</ul>`;
    } else if (sec.type === 'example') {
        inner = `
            <div class="example-box">
                <div class="example-scenario">📝 ${sec.scenario}</div>
                <div class="example-outcome">→ ${sec.outcome}</div>
            </div>`;
    } else if (sec.type === 'case_reference') {
        inner = `
            <div class="case-ref-box">
                <div class="case-ref-name">⚖️ ${sec.case}</div>
                <div class="case-ref-principle">${sec.principle}</div>
            </div>`;
    } else if (sec.type === 'flowchart') {
        const steps = sec.steps.map((s, i) =>
            `<span class="flow-step">${s}</span>${i < sec.steps.length - 1 ? '<span class="flow-arrow">→</span>' : ''}`
        ).join('');
        inner = `<div class="flowchart">${steps}</div>`;
    } else if (sec.type === 'law_section') {
        inner = `<div class="law-section-box">${sec.content}</div>`;
    }

    return `
        <div class="section-block">
            <div class="block-heading">${icon} ${sec.heading}</div>
            ${inner}
        </div>`;
}

async function completeTopic(topicId) {
    const res = await fetch(`/api/topic/${topicId}/complete`, { method: 'POST' });
    const data = await res.json();

    const btn = document.getElementById('completeBtn');
    if (btn) { btn.textContent = '✅ Completed'; btn.classList.add('done'); }

    // Update local state
    const t = allTopics.find(x => x.id === topicId);
    if (t) t.completed = true;

    showToast(`+20 pts earned! ${data.badges.length > 0 ? '🏅 New badge!' : ''}`, 'success');

    // Refresh sidebar
    const progressRes = await fetch('/api/progress');
    const progress = await progressRes.json();
    updateSidebar(progress);
}

// ── CASES ──
async function loadCases() {
    if (allCases.length === 0) {
        const res = await fetch('/api/cases');
        const data = await res.json();
        allCases = data.cases;
    }
    renderCaseList(allCases);
}

function renderCaseList(cases) {
    const list = document.getElementById('caseList');
    list.innerHTML = '';
    cases.forEach((c, i) => {
        const item = document.createElement('div');
        item.className = 'case-list-item';
        item.onclick = () => openCase(c.id);
        item.innerHTML = `
            <div class="case-num">${String(i + 1).padStart(2, '0')}</div>
            <div class="cli-body">
                <div class="cli-title">${c.title}</div>
                <div class="cli-meta">
                    <span class="level-badge ${c.difficulty}">${c.difficulty}</span>
                    <span>${c.topic}</span>
                    ${c.attempted ? '<span class="done-badge solved">✓ Attempted</span>' : ''}
                </div>
            </div>
            <div class="tli-arrow">→</div>
        `;
        list.appendChild(item);
    });
}

async function openCase(caseId) {
    const res = await fetch(`/api/case/${caseId}`);
    const data = await res.json();
    const c = data.case;

    const content = document.getElementById('caseDetailContent');
    content.innerHTML = `
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
            <div style="flex:1">
                <h2 style="font-family:var(--font-display);font-size:1.3rem;color:var(--text);margin-bottom:4px">${c.title}</h2>
                <div style="display:flex;gap:8px;align-items:center">
                    <span class="level-badge ${c.difficulty}">${c.difficulty}</span>
                    <span style="color:var(--text-2);font-size:0.8rem">${c.topic}</span>
                </div>
            </div>
        </div>
        <div class="case-scenario-box">
            <h3>📋 Scenario</h3>
            <p class="scenario-text">${c.scenario}</p>
        </div>
        <div class="case-question">${c.question}</div>
        <div id="caseOptions">
            ${c.options.map((opt, idx) => `
                <button class="option-btn" data-idx="${idx}" onclick="selectCaseOption(${idx})">${opt}</button>
            `).join('')}
        </div>
        <button class="submit-case-btn" id="caseSubmitBtn" onclick="submitCase('${caseId}')" disabled>Submit Answer →</button>
        <div id="caseJudgment"></div>
    `;

    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    document.getElementById('view-case-detail').classList.add('active');
    document.getElementById('mainContent').scrollTop = 0;
}

let selectedCaseOption = -1;

function selectCaseOption(idx) {
    selectedCaseOption = idx;
    document.querySelectorAll('.option-btn').forEach((b, i) => {
        b.classList.toggle('selected', i === idx);
    });
    document.getElementById('caseSubmitBtn').disabled = false;
}

async function submitCase(caseId) {
    if (selectedCaseOption === -1) return;

    const res = await fetch(`/api/case/${caseId}/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answer: selectedCaseOption })
    });
    const data = await res.json();

    // Disable buttons
    document.querySelectorAll('.option-btn').forEach((b, i) => {
        b.disabled = true;
        if (i === data.correct_answer) b.classList.add('correct');
        else if (i === selectedCaseOption && !data.correct) b.classList.add('wrong');
    });
    document.getElementById('caseSubmitBtn').disabled = true;

    // Show judgment
    const judgment = document.getElementById('caseJudgment');
    judgment.innerHTML = `
        <div class="judgment-box">
            <div style="font-size:1.5rem;margin-bottom:10px">${data.correct ? '✅ Correct!' : '❌ Not quite'}</div>
            <h4>⚖️ Actual Judgment</h4>
            <p class="judgment-text">${data.judgment}</p>
            <div style="margin-bottom:10px;font-size:0.8rem;color:var(--text-2)">
                <strong style="color:var(--accent)">§ Law:</strong> ${data.law_section}
            </div>
            <div class="principle-pill">💡 Principle: ${data.principle}</div>
        </div>
    `;

    // Update local case data
    const c = allCases.find(x => x.id === caseId);
    if (c) c.attempted = true;

    if (data.correct) {
        showToast('+30 pts earned! 🎉', 'success');
    } else {
        showToast('Learn from the judgment and try again!', '');
    }

    // Refresh sidebar
    const progressRes = await fetch('/api/progress');
    const progress = await progressRes.json();
    updateSidebar(progress);
}

// ── QUIZ ──
function resetQuiz() {
    document.getElementById('quizSetup').style.display = 'block';
    document.getElementById('quizPlay').style.display = 'none';
    document.getElementById('quizResult').style.display = 'none';
    quizAnswers = {};
    currentQuizIdx = 0;
    quizSubmitted = false;
}

async function startQuiz() {
    const topic = document.getElementById('quizTopic').value;
    const difficulty = document.getElementById('quizDifficulty').value;
    const count = document.getElementById('quizCount').value;

    const params = new URLSearchParams({ count });
    if (topic) params.set('topic', topic);
    if (difficulty) params.set('difficulty', difficulty);

    const res = await fetch(`/api/quiz?${params}`);
    const data = await res.json();

    if (!data.questions || data.questions.length === 0) {
        showToast('No questions found for those filters. Try a wider selection!', 'error');
        return;
    }

    quizQuestions = data.questions;
    quizAnswers = {};
    currentQuizIdx = 0;
    quizSubmitted = false;

    document.getElementById('quizSetup').style.display = 'none';
    document.getElementById('quizPlay').style.display = 'block';
    document.getElementById('quizResult').style.display = 'none';

    renderQuizQuestion();
}

function renderQuizQuestion() {
    const q = quizQuestions[currentQuizIdx];
    const total = quizQuestions.length;
    const pct = ((currentQuizIdx + 1) / total) * 100;

    document.getElementById('quizPlay').innerHTML = `
        <div class="quiz-progress-bar">
            <div class="quiz-progress-fill" style="width:${pct}%"></div>
        </div>
        <div class="quiz-q-num">Question ${currentQuizIdx + 1} of ${total}</div>
        <div class="quiz-meta">
            <span class="quiz-tag">${q.topic}</span>
            <span class="quiz-tag">${q.difficulty}</span>
            <span class="quiz-tag">${q.type}</span>
        </div>
        <div class="quiz-question">${q.question}</div>
        <div id="quizOptions">
            ${q.options.map((opt, idx) => `
                <button class="option-btn ${quizAnswers[q.id] === idx ? 'selected' : ''}"
                    onclick="selectQuizOption('${q.id}', ${idx})">${opt}</button>
            `).join('')}
        </div>
        <div class="quiz-nav">
            <button class="secondary-btn" onclick="quizNav(-1)" ${currentQuizIdx === 0 ? 'disabled' : ''}>← Prev</button>
            ${currentQuizIdx < total - 1
                ? `<button class="primary-btn" onclick="quizNav(1)">Next →</button>`
                : `<button class="primary-btn" onclick="submitQuiz()">Submit Quiz ✓</button>`
            }
        </div>
    `;
}

function selectQuizOption(qId, idx) {
    quizAnswers[qId] = idx;
    renderQuizQuestion();
}

function quizNav(dir) {
    currentQuizIdx += dir;
    currentQuizIdx = Math.max(0, Math.min(currentQuizIdx, quizQuestions.length - 1));
    renderQuizQuestion();
}

async function submitQuiz() {
    if (quizSubmitted) return;
    quizSubmitted = true;

    const unanswered = quizQuestions.filter(q => quizAnswers[q.id] === undefined).length;
    if (unanswered > 0) {
        const go = confirm(`You have ${unanswered} unanswered question(s). Submit anyway?`);
        if (!go) { quizSubmitted = false; return; }
    }

    const res = await fetch('/api/quiz/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers: quizAnswers })
    });
    const data = await res.json();

    document.getElementById('quizPlay').style.display = 'none';
    document.getElementById('quizResult').style.display = 'block';

    const stars = data.percentage >= 80 ? '⭐⭐⭐' : data.percentage >= 60 ? '⭐⭐' : '⭐';
    const grade = data.percentage >= 90 ? 'Excellent!' : data.percentage >= 70 ? 'Good job!' : data.percentage >= 50 ? 'Keep practicing.' : 'Review the concepts.';

    document.getElementById('quizResult').innerHTML = `
        <div class="result-card">
            <div class="result-stars">${stars}</div>
            <div class="result-score">${data.percentage}%</div>
            <div class="result-label">${data.score}/${data.total} correct — ${grade}</div>
            <div class="result-breakdown">+${data.points_earned} points earned</div>
        </div>
        <div class="section-heading" style="margin-bottom:14px">Answer Review</div>
        <div class="answer-review">
            ${data.results.map(r => {
                const q = quizQuestions.find(x => x.id === r.id);
                return `
                    <div class="review-item">
                        <div class="review-q">${q ? q.question : ''}</div>
                        <div class="review-result ${r.correct ? 'correct' : 'wrong'}">
                            ${r.correct ? '✅ Correct' : '❌ Incorrect'}
                            ${!r.correct && q ? ` — Correct: "${q.options[r.correct_answer]}"` : ''}
                        </div>
                        <div class="review-explanation">${r.explanation}</div>
                    </div>`;
            }).join('')}
        </div>
        <div style="display:flex;gap:12px;margin-top:24px">
            <button class="primary-btn" onclick="startQuiz()">Try Again →</button>
            <button class="secondary-btn" onclick="resetQuiz()">Change Settings</button>
        </div>
    `;

    // Refresh sidebar
    const progressRes = await fetch('/api/progress');
    const progress = await progressRes.json();
    updateSidebar(progress);
}

// ── DAILY CHALLENGE ──
async function loadDailyChallenge() {
    const res = await fetch('/api/daily-challenge');
    const data = await res.json();
    const container = document.getElementById('dailyChallengeContent');

    if (data.already_done) {
        container.innerHTML = `
            <div class="done-notice">
                <div class="done-icon">🏆</div>
                <h3>Today's Challenge Complete!</h3>
                <p style="color:var(--text-2);margin-top:6px">Come back tomorrow for a new challenge.</p>
                <p style="color:var(--text-2);margin-top:4px;font-size:0.85rem">Challenges reset at midnight.</p>
            </div>`;
        return;
    }

    const ch = data.challenge;
    timeLeft = ch.time_limit;
    clearInterval(challengeTimer);

    container.innerHTML = `
        <div class="challenge-card">
            <div class="timer-badge" id="timerBadge">⏱ <span id="timerDisplay">${formatTime(timeLeft)}</span></div>
            <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-2);margin-bottom:10px">
                📝 Case Scenario
            </div>
            <div style="font-size:0.95rem;color:var(--text);line-height:1.8;margin-bottom:20px;">${ch.scenario}</div>
            <div class="case-question">${ch.question}</div>
            <div id="challengeOptions">
                ${ch.options.map((opt, idx) => `
                    <button class="option-btn" data-idx="${idx}" onclick="selectChallengeOpt(${idx})">${opt}</button>
                `).join('')}
            </div>
            <button class="submit-case-btn" id="challengeSubmitBtn" onclick="submitChallenge('${ch.id}')" disabled>
                Submit (${ch.points} pts) →
            </button>
            <div id="challengeResult"></div>
        </div>
    `;

    // Start timer
    challengeTimer = setInterval(() => {
        timeLeft--;
        const display = document.getElementById('timerDisplay');
        if (display) display.textContent = formatTime(timeLeft);
        if (timeLeft <= 0) {
            clearInterval(challengeTimer);
            const badge = document.getElementById('timerBadge');
            if (badge) badge.style.background = 'rgba(240,96,96,0.2)';
            showToast("Time's up! See the answer below.", 'error');
            submitChallenge(ch.id, true);
        }
    }, 1000);
}

let selectedChallengeOpt = -1;

function selectChallengeOpt(idx) {
    selectedChallengeOpt = idx;
    document.querySelectorAll('#challengeOptions .option-btn').forEach((b, i) => {
        b.classList.toggle('selected', i === idx);
    });
    document.getElementById('challengeSubmitBtn').disabled = false;
}

async function submitChallenge(challengeId, timeout = false) {
    clearInterval(challengeTimer);

    const res = await fetch('/api/daily-challenge/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ challenge_id: challengeId, answer: selectedChallengeOpt })
    });
    const data = await res.json();

    document.querySelectorAll('#challengeOptions .option-btn').forEach((b, i) => {
        b.disabled = true;
        if (i === data.correct_answer) b.classList.add('correct');
        else if (i === selectedChallengeOpt && !data.correct) b.classList.add('wrong');
    });
    document.getElementById('challengeSubmitBtn').style.display = 'none';

    const result = document.getElementById('challengeResult');
    result.innerHTML = `
        <div class="judgment-box">
            <div style="font-size:1.3rem;margin-bottom:10px">${timeout ? '⏰ Time Up!' : data.correct ? '🎉 Correct!' : '❌ Incorrect'}</div>
            ${data.correct ? `<div style="color:var(--green);font-weight:700;margin-bottom:10px">+${data.points_earned} points! 🔥 ${data.streak} day streak</div>` : ''}
            <h4>⚖️ Explanation</h4>
            <p class="judgment-text">${data.explanation}</p>
        </div>
    `;

    if (data.correct) {
        showToast(`+${data.points_earned} pts! Streak: ${data.streak} days 🔥`, 'success');
        const progressRes = await fetch('/api/progress');
        const progress = await progressRes.json();
        updateSidebar(progress);
    }
}

// ── PROGRESS ──
async function loadProgress() {
    const [progressRes, leaderRes] = await Promise.all([
        fetch('/api/progress'),
        fetch('/api/leaderboard')
    ]);
    const progress = await progressRes.json();
    const leaderData = await leaderRes.json();

    const content = document.getElementById('progressContent');
    content.innerHTML = `
        <div class="progress-grid">
            <div class="progress-card">
                <h3>Topics</h3>
                <div class="progress-bar-wrap">
                    <div class="progress-bar">
                        <div class="progress-fill fill-gold" style="width:${progress.topics_percent}%"></div>
                    </div>
                    <div class="progress-pct">${progress.topics_percent}%</div>
                </div>
                <div class="progress-label">${progress.topics_completed} of ${progress.total_topics} topics completed</div>
            </div>
            <div class="progress-card">
                <h3>Case Accuracy</h3>
                <div class="progress-bar-wrap">
                    <div class="progress-bar">
                        <div class="progress-fill fill-green" style="width:${progress.case_accuracy}%"></div>
                    </div>
                    <div class="progress-pct">${progress.case_accuracy}%</div>
                </div>
                <div class="progress-label">${progress.cases_attempted} of ${progress.total_cases} cases attempted</div>
            </div>
            <div class="progress-card">
                <h3>Quiz Performance</h3>
                <div class="progress-bar-wrap">
                    <div class="progress-bar">
                        <div class="progress-fill fill-blue" style="width:${progress.avg_quiz_score}%"></div>
                    </div>
                    <div class="progress-pct">${progress.avg_quiz_score}%</div>
                </div>
                <div class="progress-label">Average quiz score across ${progress.quiz_scores.length} attempt(s)</div>
                ${progress.quiz_scores.map(s => `
                    <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:0.78rem;color:var(--text-2)">
                        <span>${s.date}</span>
                        <span>${s.correct}/${s.total} — ${s.score}%</span>
                    </div>`).join('')}
            </div>
            <div class="progress-card">
                <h3>Overall Stats</h3>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:4px">
                    ${[
                        ['🔥', 'Streak', progress.streak + ' days'],
                        ['⭐', 'Points', progress.points + ' pts'],
                        ['🎓', 'Level', progress.level],
                        ['🏅', 'Badges', progress.badges.length + ' earned']
                    ].map(([icon, label, val]) => `
                        <div style="background:var(--bg-card2);border-radius:10px;padding:12px;text-align:center">
                            <div style="font-size:1.4rem">${icon}</div>
                            <div style="font-size:1rem;font-weight:700;color:var(--accent);margin:4px 0">${val}</div>
                            <div style="font-size:0.72rem;color:var(--text-2)">${label}</div>
                        </div>`).join('')}
                </div>
            </div>
        </div>

        <div class="section-heading">Badges Earned</div>
        <div class="badges-row" style="margin-bottom:28px">
            ${progress.badges.length === 0
                ? '<span class="no-badges">No badges yet. Complete topics and quizzes! 🏅</span>'
                : progress.badges.map(b => `
                    <div class="badge-pill">
                        <span class="badge-icon">${badgeIcon(b)}</span>${b}
                    </div>`).join('')}
        </div>

        <div class="section-heading">Leaderboard</div>
        <div style="background:var(--bg-card);border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;">
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Student</th>
                        <th>Points</th>
                        <th>Level</th>
                        <th>Streak</th>
                    </tr>
                </thead>
                <tbody>
                    ${leaderData.leaderboard.map(entry => `
                        <tr class="${entry.name === 'You' ? 'you' : ''}">
                            <td>${rankMedal(entry.rank)}</td>
                            <td>${entry.name === 'You' ? '👤 You' : entry.name}</td>
                            <td>${entry.points}</td>
                            <td><span class="level-badge ${entry.level}">${entry.level}</span></td>
                            <td>🔥 ${entry.streak}</td>
                        </tr>`).join('')}
                </tbody>
            </table>
        </div>
    `;
}

// ── HELPERS ──
function buildTopicCard(t) {
    const div = document.createElement('div');
    div.className = `topic-card ${t.completed ? 'done' : ''}`;
    div.onclick = () => openTopic(t.id);
    div.innerHTML = `
        ${t.completed ? '<div class="done-check">✅</div>' : ''}
        <div class="tc-icon">${t.icon}</div>
        <div class="tc-title">${t.title}</div>
        <div class="tc-meta">
            <span class="level-badge ${t.level}">${t.level}</span>
            <span>⏱ ${t.estimated_time}</span>
        </div>
    `;
    return div;
}

function formatTime(s) {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${String(sec).padStart(2, '0')}`;
}

function badgeIcon(name) {
    const icons = {
        'First Step': '🎯',
        'Quiz Master': '🧩',
        'Case Solver': '⚖️',
        'Streak Week': '🔥',
        'Night Owl Jurist': '🦉',
        'Digital Rights Defender': '🛡️',
        'Legal Eagle': '🦅',
        'Contract Champion': '📜',
        'Perfect Score': '💯'
    };
    return icons[name] || '🏅';
}

function rankMedal(rank) {
    return rank === 1 ? '🥇' : rank === 2 ? '🥈' : rank === 3 ? '🥉' : `#${rank}`;
}

function showToast(msg, type = '') {
    const toast = document.getElementById('toast');
    toast.textContent = msg;
    toast.className = `toast ${type} show`;
    setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── THEME ──
function setTheme(name) {
    document.body.classList.remove('theme-obsidian', 'theme-slate', 'theme-paper');
    document.body.classList.add(`theme-${name}`);
    localStorage.setItem('lmStudentTheme', name);
}
function loadTheme() {
    const t = localStorage.getItem('lmStudentTheme');
    if (t) setTheme(t);
}

// ── MOBILE NAV ──
function initMobileNav() {
    document.getElementById('mobileMenuBtn')?.addEventListener('click', () => {
        document.getElementById('sidebar').classList.add('open');
        document.getElementById('sidebarOverlay').classList.add('active');
        document.body.style.overflow = 'hidden';
    });
    document.getElementById('mobileCloseBtn')?.addEventListener('click', closeMobileSidebar);
    document.getElementById('sidebarOverlay')?.addEventListener('click', closeMobileSidebar);
}
function closeMobileSidebar() {
    if (window.innerWidth <= 768) {
        document.getElementById('sidebar').classList.remove('open');
        document.getElementById('sidebarOverlay').classList.remove('active');
        document.body.style.overflow = '';
    }
}