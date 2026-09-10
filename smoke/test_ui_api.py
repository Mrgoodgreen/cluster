"""Search/filter pagination uses all tasks and never changes the queue."""
import importlib
import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from urllib.parse import urlencode,urlsplit
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'manager'))
from app.models import Base,Task
with patch('app.models.init_db'):
    main=importlib.import_module('app.main')


class ListingTests(unittest.TestCase):
    def setUp(self):
        self.engine=create_engine('sqlite://',connect_args={'check_same_thread':False},poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db=Session(self.engine)
        statuses=['pending','processing','completed','error','cancelled']
        for i in range(25):
            name='needle' if i==0 else 'percent_%_sample' if i==1 else f'scan-{i}'
            self.db.add(Task(uid=f'uid-{i}',status=statuses[i%5],input_path=f'storage/{name}',output_path=f'output/{name}'))
        self.db.commit()
        def override():yield self.db
        main.app.dependency_overrides[main.get_db]=override
        self.client=SimpleNamespace(get=self.get)
    def get(self,url,params=None):
        parsed=urlsplit(url)
        query=urlencode(params) if params is not None else parsed.query
        messages=[]
        async def receive():return {'type':'http.request','body':b'','more_body':False}
        async def send(message):messages.append(message)
        scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'GET',
               'path':parsed.path,'raw_path':parsed.path.encode(),'query_string':query.encode(),
               'scheme':'http','headers':[],'server':('test',80),'client':('127.0.0.1',1234),'root_path':''}
        asyncio.run(main.app(scope,receive,send))
        status=next(m['status'] for m in messages if m['type']=='http.response.start')
        body=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
        return SimpleNamespace(status_code=status,json=lambda:json.loads(body))
    def tearDown(self):
        main.app.dependency_overrides.clear();self.db.close();self.engine.dispose()
    def test_original_pagination_and_global_counts(self):
        data=self.client.get('/api/tasks?page=2&page_size=20').json()
        self.assertEqual(data['total'],25);self.assertEqual(len(data['items']),5)
        self.assertEqual(data['status_counts'],dict.fromkeys(['pending','processing','completed','error','cancelled'],5))
    def test_search_reaches_task_outside_first_page(self):
        data=self.client.get('/api/tasks?q=needle&status=pending').json()
        self.assertEqual(data['total'],1);self.assertEqual(data['items'][0]['id'],1)
        self.assertEqual(sum(data['status_counts'].values()),25)
    def test_literal_sql_wildcards(self):
        data=self.client.get('/api/tasks',params={'q':'%_'}).json()
        self.assertEqual(data['total'],1)
        self.assertEqual(data['items'][0]['id'],2)
    def test_invalid_filters_rejected(self):
        for params in ({'status':'made-up'},{'page_size':101},{'q':'a'*201}):
            self.assertEqual(self.client.get('/api/tasks',params=params).status_code,422)
        self.assertEqual(self.client.get('/api/tasks').json()['total'],25)

if __name__=='__main__':unittest.main()
