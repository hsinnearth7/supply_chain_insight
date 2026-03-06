import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, statSync } from 'fs';
import { join } from 'path';

const BASE_URL = 'http://localhost:8000';
const API_KEY = 'dev-key-change-me';
const SCREENSHOT_DIR = './screenshots';

// Sidebar link text -> screenshot filename
const pages = [
  { linkText: 'Dashboard', name: 'dashboard', title: 'Dashboard' },
  { linkText: 'Upload', name: 'upload', title: 'Upload & Run' },
  { linkText: 'Statistics', name: 'stats', title: 'Statistics' },
  { linkText: 'Supply Chain', name: 'supply-chain', title: 'Supply Chain' },
  { linkText: 'ML', name: 'ml', title: 'ML / AI' },
  { linkText: 'RL', name: 'rl', title: 'RL Optimization' },
  { linkText: 'History', name: 'history', title: 'History' },
];

async function waitForServer(url, maxWait = 30000) {
  const start = Date.now();
  while (Date.now() - start < maxWait) {
    try {
      const res = await fetch(`${url}/api/health`);
      if (res.ok) return true;
    } catch {}
    await new Promise(r => setTimeout(r, 1000));
  }
  throw new Error(`Server not ready after ${maxWait}ms`);
}

async function injectApiKey(page) {
  // Intercept all /api/ requests and inject the API key header
  await page.route('**/api/**', async (route) => {
    const headers = {
      ...route.request().headers(),
      'X-API-Key': API_KEY,
    };
    await route.continue({ headers });
  });
}

async function main() {
  mkdirSync(SCREENSHOT_DIR, { recursive: true });

  console.log('Waiting for server...');
  await waitForServer(BASE_URL);
  console.log('Server is ready!');

  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const results = [];

  // === Light Mode ===
  const tab = await context.newPage();
  await injectApiKey(tab);

  // Load the SPA once
  console.log('Loading SPA...');
  await tab.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 20000 });
  await tab.waitForTimeout(3000);

  for (const page of pages) {
    console.log(`Capturing: ${page.title}`);
    let success = false;

    for (let attempt = 1; attempt <= 3; attempt++) {
      try {
        // Click sidebar link to navigate (client-side routing)
        if (page.linkText !== 'Dashboard') {
          const link = tab.locator(`nav a, aside a, [class*="sidebar"] a, a`).filter({ hasText: page.linkText }).first();
          await link.click({ timeout: 5000 });
          await tab.waitForTimeout(3000);
        }

        const filePath = join(SCREENSHOT_DIR, `${page.name}.png`);
        await tab.screenshot({ path: filePath, fullPage: true });
        const size = statSync(filePath).size;

        if (size > 10240) {
          console.log(`  OK: ${page.name}.png (${(size / 1024).toFixed(1)} KB)`);
          results.push({ page: page.title, file: `${page.name}.png`, status: 'PASS', size });
          success = true;
          break;
        } else {
          console.log(`  WARN: too small (${size} bytes), attempt ${attempt}`);
          await tab.waitForTimeout(attempt * 2000);
        }
      } catch (err) {
        console.log(`  Attempt ${attempt} failed: ${err.message.slice(0, 100)}`);
        await tab.waitForTimeout(attempt * 3000);
      }
    }

    if (!success) {
      // Fallback: take whatever is on screen
      const filePath = join(SCREENSHOT_DIR, `${page.name}.png`);
      try {
        await tab.screenshot({ path: filePath, fullPage: true });
        const size = statSync(filePath).size;
        results.push({ page: page.title, file: `${page.name}.png`, status: 'WARN', size });
        console.log(`  WARN: saved anyway (${(size / 1024).toFixed(1)} KB)`);
      } catch {
        results.push({ page: page.title, file: `${page.name}.png`, status: 'FAIL', size: 0 });
      }
    }
  }
  await tab.close();

  // === Dark Mode ===
  console.log('\n--- Dark Mode ---');
  const darkTab = await context.newPage();
  await injectApiKey(darkTab);

  await darkTab.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 20000 });
  await darkTab.waitForTimeout(2000);

  // Click the Dark mode toggle button
  try {
    const darkBtn = darkTab.locator('button, [role="button"]').filter({ hasText: 'Dark' }).first();
    await darkBtn.click({ timeout: 5000 });
    await darkTab.waitForTimeout(2000);
  } catch (err) {
    console.log('  Could not find dark toggle, forcing via JS');
    await darkTab.evaluate(() => document.documentElement.classList.add('dark'));
  }

  // Dashboard dark
  {
    const filePath = join(SCREENSHOT_DIR, 'dashboard-dark.png');
    await darkTab.screenshot({ path: filePath, fullPage: true });
    const size = statSync(filePath).size;
    console.log(`Capturing: Dashboard (Dark) - ${(size / 1024).toFixed(1)} KB`);
    results.push({ page: 'Dashboard (Dark)', file: 'dashboard-dark.png', status: size > 10240 ? 'PASS' : 'WARN', size });
  }

  // Stats dark - click Statistics link
  try {
    const statsLink = darkTab.locator('a').filter({ hasText: 'Statistics' }).first();
    await statsLink.click({ timeout: 5000 });
    await darkTab.waitForTimeout(3000);
    const filePath = join(SCREENSHOT_DIR, 'stats-dark.png');
    await darkTab.screenshot({ path: filePath, fullPage: true });
    const size = statSync(filePath).size;
    console.log(`Capturing: Stats (Dark) - ${(size / 1024).toFixed(1)} KB`);
    results.push({ page: 'Stats (Dark)', file: 'stats-dark.png', status: size > 10240 ? 'PASS' : 'WARN', size });
  } catch (err) {
    console.log(`  Stats dark failed: ${err.message.slice(0, 80)}`);
    results.push({ page: 'Stats (Dark)', file: 'stats-dark.png', status: 'FAIL', size: 0 });
  }

  await darkTab.close();
  await browser.close();

  // Generate README
  const passed = results.filter(r => r.status === 'PASS').length;
  const warned = results.filter(r => r.status === 'WARN').length;
  const total = results.length;

  let md = `# ChainInsight Dashboard Screenshots\n\n`;
  md += `Captured: ${new Date().toISOString().slice(0, 10)}\n`;
  md += `Results: ${passed} passed, ${warned} warnings, ${total - passed - warned} failed / ${total} total\n\n`;
  md += `## Light Mode\n\n`;
  md += `| Page | Screenshot | Status | Size |\n`;
  md += `|------|-----------|--------|------|\n`;
  for (const r of results.filter(r => !r.file.includes('dark'))) {
    const sizeStr = r.size > 0 ? `${(r.size / 1024).toFixed(1)} KB` : '-';
    const img = r.status !== 'FAIL' ? `![${r.page}](${r.file})` : '(failed)';
    md += `| ${r.page} | ${img} | ${r.status} | ${sizeStr} |\n`;
  }
  md += `\n## Dark Mode\n\n`;
  md += `| Page | Screenshot | Status | Size |\n`;
  md += `|------|-----------|--------|------|\n`;
  for (const r of results.filter(r => r.file.includes('dark'))) {
    const sizeStr = r.size > 0 ? `${(r.size / 1024).toFixed(1)} KB` : '-';
    const img = r.status !== 'FAIL' ? `![${r.page}](${r.file})` : '(failed)';
    md += `| ${r.page} | ${img} | ${r.status} | ${sizeStr} |\n`;
  }

  writeFileSync(join(SCREENSHOT_DIR, 'README.md'), md);
  console.log(`\nDone! ${passed} passed, ${warned} warnings out of ${total} screenshots.`);
  console.log(`Summary: ${SCREENSHOT_DIR}/README.md`);
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
