// Exercise the production React bundle without starting a manager or web server.
// API responses are isolated fixtures. Set PLAYWRIGHT_PATH if not installed locally.
const {chromium} = require(process.env.PLAYWRIGHT_PATH || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const build = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
fs.mkdirSync(output,{recursive:true});
const statuses=['pending','processing','completed','error','cancelled'];
function task(id,status,name=`project_${id}`) {
  return {id,uid:`task-${id}`,status,input_path:`172.16.16.126/ishod/${name}`,output_path:`172.16.16.126/clasify/${name}`,created_at:'2026-09-09T06:00:00',finished_at:status==='completed'?'2026-09-09T07:00:00Z':null,error_message:null,subtask_total:3,subtask_done:status==='completed'?3:1,
    subtasks:[0,1,2].map(i=>({id:id*10+i,filename:`${name}_${i}.las`,relative_path:`${name}_${i}.las`,status:status==='completed'?'success':i===0?'success':i===1&&status==='processing'?'processing':status==='error'?'error':'pending',progress:i===0?100:35,worker_id:i<2?'gpu-node-01':null,started_at:i<2?'2026-09-09T06:05:00':null,finished_at:i===0?'2026-09-09T06:10:00':null,error_message:status==='error'?'Ошибка записи SMB':null,log_text:i===0?'Готово <script>window.bad=true</script>':'GPU: обработка'}))};
}
let tasks=Array.from({length:42},(_,i)=>task(i+1,i+1===42?'processing':i+1===39?'error':i+1===40?'pending':'completed',i+1===42?'3dit_01208sm6':`project_${i+1}`));
const folders={'': ['172.16.16.126'],'172.16.16.126':['ishod','clasify'],'172.16.16.126/ishod':['3dit_01208sm6'],'172.16.16.126/ishod/3dit_01208sm6':[],'172.16.16.126/clasify':[]};
const actions=[], calls=[], errors=[];let failList=false, failCreate=false, failTree=false;
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  try {
    const page=await browser.newPage({viewport:{width:1440,height:1060}});
    async function snapshot(name) {
      await page.waitForFunction(()=>!document.querySelector('.ant-spin-blur'));
      await page.screenshot({path:path.join(output,`${name}.png`),fullPage:true,animations:'disabled',style:'.ant-message{visibility:hidden} *{transition:none!important}'});
    }
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/*',async route=>{
      const req=route.request(), url=new URL(req.url()), method=req.method(), p=url.pathname;
      if(url.origin!=='http://tls.test') {errors.push(`Unexpected network: ${url}`);return route.abort();}
      const json=(body,status=200)=>route.fulfill({status,contentType:'application/json',body:JSON.stringify(body)});
      if(p.startsWith('/api/')) {
        calls.push(p+url.search);
        if(p==='/api/health')return json({ok:true,storage_roots:['172.16.16.126']});
        if(p==='/api/storage/tree') {
          if(failTree)return json({detail:'SMB недоступен'},503);
          const dir=url.searchParams.get('path')||'';
          return json({nodes:(folders[dir]||[]).map(name=>({title:name,path:dir?`${dir}/${name}`:name}))});
        }
        if(p==='/api/storage/mkdir') {
          const body=req.postDataJSON();actions.push({action:'mkdir',...body});
          const existed=Object.hasOwn(folders,body.path);folders[body.path]??=[];
          const slash=body.path.lastIndexOf('/'),parent=body.path.slice(0,slash),name=body.path.slice(slash+1);
          folders[parent]??=[];if(!folders[parent].includes(name))folders[parent].push(name);
          return json({path:body.path,created:!existed,existed});
        }
        if(p==='/api/tasks'&&method==='GET') {
          if(failList)return json({detail:'Менеджер временно недоступен'},503);
          const counts={};for(const t of tasks)counts[t.status]=(counts[t.status]||0)+1;
          const q=(url.searchParams.get('q')||'').toLowerCase(),status=url.searchParams.get('status');
          const filtered=tasks.filter(t=>(!status||t.status===status)&&(!q||`${t.input_path} ${t.uid} ${t.id}`.toLowerCase().includes(q))).sort((a,b)=>b.id-a.id);
          const current=Number(url.searchParams.get('page')||1),size=Number(url.searchParams.get('page_size')||20);
          return json({items:filtered.slice((current-1)*size,current*size).map(({subtasks,...t})=>t),total:filtered.length,page:current,page_size:size,status_counts:counts});
        }
        if(p==='/api/tasks'&&method==='POST') {
          if(failCreate)return json({detail:'Нет LAS в выбранной папке'},400);
          const body=req.postDataJSON(),t=task(Math.max(...tasks.map(t=>t.id))+1,'pending');Object.assign(t,body);tasks.push(t);actions.push({action:'create',...body});return json(t);
        }
        const match=p.match(/^\/api\/tasks\/(\d+)(?:\/(cancel|restart))?$/);
        if(match) {
          const t=tasks.find(t=>t.id===Number(match[1]));if(!t)return json({detail:'Not found'},404);
          if(match[2]==='cancel') {t.status='cancelled';t.subtasks.forEach(s=>{if(['processing','pending'].includes(s.status))s.status='cancelled';});actions.push({action:'cancel',id:t.id});return json(t);}
          if(match[2]==='restart') {assert.notEqual(t.status,'processing');const n=task(Math.max(...tasks.map(t=>t.id))+1,'pending');n.input_path=t.input_path;n.output_path=t.output_path;tasks.push(n);actions.push({action:'restart',id:t.id});return json(n);}
          return json(t);
        }
        errors.push(`Unknown API ${method} ${p}`);return json({},404);
      }
      const file=p.startsWith('/assets/')?path.join(build,p.slice(1)):path.join(build,'index.html');
      assert.ok(file.startsWith(build+path.sep));
      return route.fulfill({path:file,contentType:file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':'text/html'});
    });
    await page.goto('http://tls.test/tasks');
    await page.getByRole('heading',{name:'Задачи',exact:true}).waitFor();
    await page.locator('.ant-table-tbody tr[data-row-key]').first().waitFor();
    assert.equal(await page.locator('.ant-table-tbody tr[data-row-key]').count(),20);
    console.log('Fonts:',await page.evaluate(()=>[getComputedStyle(document.body).fontFamily,getComputedStyle(document.querySelector('h1')).fontFamily]));
    await snapshot('tasks');
    await page.locator('.ant-pagination-item-2').click();
    await page.locator('tr[data-row-key="22"]').waitFor();
    await page.getByRole('button',{name:'Ошибки',exact:true}).click();
    await page.locator('tr[data-row-key="39"]').waitFor();assert.equal(await page.locator('.ant-table-tbody tr[data-row-key]').count(),1);
    await page.getByRole('button',{name:'Все задачи',exact:true}).click();
    await page.getByRole('textbox',{name:'Поиск задачи'}).fill('3dit_01208sm6');
    await page.waitForFunction(()=>document.querySelectorAll('.ant-table-tbody tr[data-row-key]').length===1);
    assert.ok(calls.some(c=>c.includes('q=3dit_01208sm6')));
    await page.locator('a.taskname').click();
    await page.getByRole('heading',{name:'Задача #42',exact:true}).waitFor();
    assert.equal(await page.getByRole('button',{name:'Перезапустить',exact:true}).isDisabled(),true);
    assert.match(await page.locator('.details-grid').innerText(),/09\.09\.2026 09:00/);
    tasks.find(t=>t.id===42).subtasks[1].log_text='GPU: новый прогресс';
    await page.waitForFunction(()=>document.querySelector('.log')?.textContent.includes('новый прогресс'));
    await page.locator('tr[data-row-key="420"] button.filename').click();
    assert.match(await page.locator('.log').innerText(),/<script>/);assert.equal(await page.evaluate(()=>window.bad),undefined);
    await page.locator('tr[data-row-key="421"] button.filename').click();
    await page.waitForFunction(()=>document.querySelector('.log')?.textContent.includes('новый прогресс'));
    await snapshot('detail');
    await page.getByRole('button',{name:'Отменить',exact:true}).click();
    await page.getByRole('button',{name:'Отменить задачу',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('.detailtop .status')?.textContent==='Отменена');
    assert.equal(await page.getByRole('button',{name:'Перезапустить',exact:true}).isEnabled(),true);
    const beforeTerminal=calls.filter(c=>c==='/api/tasks/42').length;
    await page.waitForTimeout(2800);
    assert.equal(calls.filter(c=>c==='/api/tasks/42').length,beforeTerminal);
    await page.getByRole('button',{name:'Перезапустить',exact:true}).click();
    await page.getByRole('heading',{name:'Задача #43',exact:true}).waitFor();
    assert.equal(tasks.find(t=>t.id===43).input_path,tasks.find(t=>t.id===42).input_path);
    await page.getByRole('link',{name:'Создать задачу',exact:true}).click();
    await page.locator('#inputPath').fill('172.16.16.126/ishod/3dit_01208sm6');
    assert.equal(await page.locator('#outputPath').inputValue(),'172.16.16.126/ishod/3dit_01208sm6_classified');
    await page.locator('#inputPath').fill('');assert.equal(await page.locator('#outputPath').inputValue(),'');
    await page.getByRole('button',{name:'Выбрать папку',exact:true}).click();
    await page.locator('.folder').filter({hasText:'172.16.16.126'}).click();
    await page.locator('.folder').filter({hasText:'ishod'}).click();
    await page.getByRole('button',{name:'Новая папка',exact:true}).click();
    await page.getByRole('textbox',{name:'Имя новой папки'}).fill('new_project');
    await page.getByRole('button',{name:'Создать папку',exact:true}).click();
    await page.locator('.folder').filter({hasText:'new_project'}).waitFor();
    assert.ok(actions.some(a=>a.action==='mkdir'&&a.path==='172.16.16.126/ishod/new_project'));
    await snapshot('folders');
    await page.locator('.folder').filter({hasText:'3dit_01208sm6'}).click();
    await page.getByRole('button',{name:'Выбрать эту папку',exact:true}).click();
    await page.getByRole('dialog').waitFor({state:'hidden'});
    assert.equal(await page.locator('#inputPath').inputValue(),'172.16.16.126/ishod/3dit_01208sm6');
    await snapshot('create');
    await page.getByRole('switch',{name:'Автоматическая выходная папка'}).click();
    await page.locator('#outputPath').fill('172.16.16.126/clasify/manual');
    failCreate=true;
    await page.getByRole('button',{name:'Поставить в очередь'}).click();
    await page.getByText('Нет LAS в выбранной папке',{exact:true}).waitFor();
    assert.equal(await page.locator('#outputPath').inputValue(),'172.16.16.126/clasify/manual');
    failCreate=false;
    await page.getByRole('button',{name:'Поставить в очередь'}).click();
    await page.getByRole('heading',{name:'Задача #44',exact:true}).waitFor();
    assert.deepEqual(actions.filter(a=>a.action==='create').at(-1),{action:'create',input_path:'172.16.16.126/ishod/3dit_01208sm6',output_path:'172.16.16.126/clasify/manual'});
    await page.getByRole('link',{name:'Задачи',exact:true}).click();
    failList=true;await page.getByRole('button',{name:'Обновить задачи',exact:true}).click();
    await page.getByText('Менеджер временно недоступен',{exact:true}).waitFor();
    failList=false;await page.getByRole('button',{name:'Повторить',exact:true}).click();
    await page.locator('tr[data-row-key="44"]').waitFor();
    await page.setViewportSize({width:390,height:844});
    await snapshot('mobile');
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Page overflows mobile viewport');
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(output,'checks.json'),JSON.stringify({passed:true,api_actions:actions,requests:calls.length,page_errors:errors,screenshots:['tasks','detail','folders','create','mobile']},null,2));
    console.log('Production UI checks passed: API wiring, pagination/search/filter, logs, cancel/restart, folders, creation/errors, mobile; no server started.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
