const { JSDOM } = require('jsdom');
const BASE = process.env.BASE || 'http://127.0.0.1:8123';
const sleep = ms => new Promise(r => setTimeout(r, ms));
let fails = 0;
const ok = (c, m) => { console.log((c ? 'PASS ' : 'FAIL ') + m); if (!c) fails++; };

(async () => {
  const errors = [];
  const dom = await JSDOM.fromURL(BASE + '/', {
    runScripts: 'dangerously', pretendToBeVisual: true,
    beforeParse(w) {
      w.fetch = (u, o = {}) => { const { signal, ...rest } = o; return fetch(new URL(u, BASE).href, rest); };
      w.addEventListener('error', e => errors.push(e.message));
      w.scrollTo = () => {};
      w.URL.createObjectURL = () => 'blob:x';
    }
  });
  const w = dom.window, d = w.document, $ = s => d.querySelector(s);
  const click = el => el.dispatchEvent(new w.MouseEvent('click', { bubbles: true }));
  const type = (el, v) => { el.value = v; el.dispatchEvent(new w.Event('input', { bubbles: true })); };
  const text = () => $('#main').textContent;

  await sleep(1500);
  ok(/Model online/.test($('#api-st').textContent), 'status chip shows "Model online": ' + $('#api-st').textContent);

  // load demo data from Settings
  click($('nav button[data-p="settings"]')); await sleep(100);
  ok(/Connected/.test(text()) && /Hold-out accuracy 97%/.test(text()), 'Settings shows connection + model metrics');
  click($('[data-act="demo"]')); await sleep(2500);
  ok(/Spending outlook/.test(text()), 'Overview shows server "Spending outlook" card');
  ok(/Safe daily spend/.test(text()), 'capacity numbers rendered');

  click($('nav button[data-p="insights"]')); await sleep(200);
  const ins = text();
  ok(/Netflix/i.test(ins) && /next expected/.test(ins), 'Insights lists recurring Netflix with next expected date');
  ok(/Jio/i.test(ins), 'Insights lists recurring Jio recharge');
  ok(/detectors agree/.test(ins) || /No|Nothing unusual/.test(ins), 'Unusual-spending section is model-driven or empty-state');
  ok(/laptop accessories/i.test(ins) || /Apollo hospital/i.test(ins), 'Insights flags a demo outlier (laptop accessories / hospital visit)');
  console.log('   anomalies text:', (ins.match(/Unusual spending.*?(Flagged|Nothing)/s) || [''])[0].slice(0, 220).replace(/\s+/g, ' '));

  click($('nav button[data-p="budgets"]')); await sleep(200);
  ok(/Suggested starting budgets/.test(text()) || /Spending outlook/.test(text()), 'Budgets shows outlook/suggestions');

  // add transaction: model suggestion appears while typing
  click($('nav button[data-p="transactions"]')); await sleep(100);
  click($('[data-act="add-tx"]')); await sleep(100);
  const f = $('form[data-form="tx"]');
  type(f.elements.amount, '420');
  type(f.elements.description, 'UPI-ZOMATO-zomato@hdfcbank-998877-Payment');
  await sleep(900);
  const sug = $('#sug').textContent;
  ok(/Suggested: Food \(model, \d+% confident/.test(sug), 'live model suggestion: ' + sug);
  ok(f.elements.category.value === 'Food', 'category select auto-set to Food');

  // user overrides the model -> feedback stored
  f.elements.category.value = 'Shopping'; f.elements.category.dispatchEvent(new w.Event('input', { bubbles: true }));
  f.dispatchEvent(new w.Event('submit', { bubbles: true, cancelable: true }));
  await sleep(1200);
  const h = await (await fetch(BASE + '/api/health')).json();
  ok(h.model.corrections >= 1, 'override sent to server as a correction (corrections=' + h.model.corrections + ')');

  // CSV import without a category column
  click($('nav button[data-p="settings"]')); await sleep(100);
  const inp = $('[data-imp]');
  const csv = 'Date,Narration,Amount\n2026-10-05 12:30,Swiggy order,350\n2026-10-06 09:10,Uber ride,180\n2026-10-07 18:00,Apollo Pharmacy,540\n2026-10-08 08:00,bad row,abc\n';
  Object.defineProperty(inp, 'files', { value: [{ text: async () => csv }] });
  inp.dispatchEvent(new w.Event('change', { bubbles: true }));
  await sleep(1500);
  const toastText = $('#toast').textContent;
  ok(/Imported 3 rows/.test(toastText) && /categorised by the model/.test(toastText) && /skipped 1/.test(toastText), 'CSV import w/o category: ' + toastText);
  click($('nav button[data-p="transactions"]')); await sleep(100);
  const list = $('#txl').textContent;
  ok(/Apollo Pharmacy.*Healthcare/s.test(list.replace(/\s+/g, ' ')), 'imported Apollo row got category Healthcare');

  // offline fallback
  const real = w.fetch; w.fetch = () => Promise.reject(new Error('offline'));
  await w.ping(); await sleep(100);
  ok(/Offline mode/.test($('#api-st').textContent), 'status flips to Offline mode when server unreachable');
  click($('nav button[data-p="insights"]')); await sleep(200);
  ok(!/is not defined|undefined/.test(text()) && text().length > 200, 'Insights still renders offline (local rules)');
  w.fetch = real; await w.ping(); await sleep(100);
  ok(/Model online/.test($('#api-st').textContent), 'status recovers to online');

  ok(errors.length === 0, 'no uncaught JS errors ' + JSON.stringify(errors));
  console.log(fails ? `\n${fails} FAILED` : '\nALL E2E CHECKS PASSED');
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
