// ── Navbar scroll effect
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 20);
  updateActiveNav();
}, { passive: true });

// ── Mobile burger menu
const burger = document.getElementById('burger');
const navLinks = document.getElementById('navLinks');
burger.addEventListener('click', () => navLinks.classList.toggle('open'));
navLinks.querySelectorAll('a').forEach(a => a.addEventListener('click', () => navLinks.classList.remove('open')));

// ── Active nav link tracking
const sections = document.querySelectorAll('section[id]');
const navAnchors = document.querySelectorAll('.nav-links a');
function updateActiveNav() {
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 100) current = s.id;
  });
  navAnchors.forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === '#' + current);
  });
}

// ── Scroll-reveal with stagger
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    const el = e.target;
    const delay = el.dataset.delay || 0;
    setTimeout(() => {
      el.style.opacity = '1';
      el.style.transform = 'translateY(0)';
    }, delay);
    revealObserver.unobserve(el);
  });
}, { threshold: 0.12 });

function setupReveal() {
  // Cards with stagger in grids
  document.querySelectorAll('.grid-2 > *, .grid-3 > *').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(28px)';
    el.style.transition = `opacity .55s ease, transform .55s ease`;
    el.dataset.delay = (i % 3) * 120;
    revealObserver.observe(el);
  });

  // Individual blocks
  document.querySelectorAll('.sentinel-block, .flowchart-wrap, .section-title, .section-divider, table, footer').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(24px)';
    el.style.transition = 'opacity .6s ease, transform .6s ease';
    el.dataset.delay = 0;
    revealObserver.observe(el);
  });

  // Sentinel steps slide in from left sequentially
  document.querySelectorAll('.sentinel-step').forEach((el, i) => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(-20px)';
    el.style.transition = 'opacity .5s ease, transform .5s ease';
    el.dataset.delay = i * 100;
    revealObserver.observe(el);
  });
}
setupReveal();

// ── Mouse-tracking glow on cards
document.querySelectorAll('.card').forEach(card => {
  card.addEventListener('mousemove', e => {
    const r = card.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width * 100).toFixed(1);
    const y = ((e.clientY - r.top) / r.height * 100).toFixed(1);
    card.style.setProperty('--mx', x + '%');
    card.style.setProperty('--my', y + '%');
  });
});

// ── Count-up animation for stat numbers
function countUp(el, target, suffix, duration = 1200) {
  const isFloat = target % 1 !== 0;
  const start = performance.now();
  function frame(now) {
    const p = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    const val = isFloat ? (eased * target).toFixed(1) : Math.round(eased * target);
    el.textContent = val + suffix;
    if (p < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

const countObserver = new IntersectionObserver((entries) => {
  entries.forEach(e => {
    if (!e.isIntersecting) return;
    const el = e.target;
    const raw = el.dataset.count;
    const suffix = el.dataset.suffix || '';
    if (raw === undefined) return;
    countUp(el, parseFloat(raw), suffix);
    countObserver.unobserve(el);
  });
}, { threshold: 0.5 });

document.querySelectorAll('[data-count]').forEach(el => countObserver.observe(el));

// ── Smooth section reveal on page load (above fold)
window.addEventListener('load', () => {
  document.querySelectorAll('.hero-inner > *').forEach((el, i) => {
    el.style.animationFillMode = 'both';
  });
  updateActiveNav();
});
