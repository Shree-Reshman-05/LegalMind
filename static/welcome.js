let selectedTheme = localStorage.getItem('legalMindTheme') || 'obsidian';

// Apply saved theme on load
document.body.className = `theme-${selectedTheme}`;

function selectTheme(theme) {
  selectedTheme = theme;

  document.querySelectorAll('.theme-circle')
    .forEach(btn => btn.classList.remove('active'));

  // Mark active button safely
  const btn = document.querySelector(`.theme-circle.${theme === 'paper' ? 'light' : theme}`);
  if (btn) btn.classList.add('active');

  document.body.className = `theme-${theme}`;
  localStorage.setItem('legalMindTheme', theme);
}

function enterApp() {
  window.location.href = "/chatbot";
}
