import requests
BASE = 'http://127.0.0.1:8000/api'

def dump(resp, label):
    text = resp.text
    try:
        data = resp.json()
    except Exception:
        data = None
    print(label, 'status=', resp.status_code, 'json=', data, 'text=', text[:200])
    return data

resp = requests.get(BASE + '/', timeout=30)
dump(resp, 'ROOT')
resp = requests.get(BASE + '/openapi.json', timeout=30)
dump(resp, 'OPENAPI')

resp = requests.post(BASE + '/auth/login', json={'email': 'patient1@example.com', 'password': 'password'}, timeout=30)
pat_login = dump(resp, 'PAT LOGIN')
if not pat_login or 'access_token' not in pat_login:
    raise SystemExit('Patient login failed')
pat_auth = {'Authorization': f"Bearer {pat_login['access_token']}"}

resp = requests.get(BASE + '/auth/me', headers=pat_auth, timeout=30)
dump(resp, 'PAT ME')
resp = requests.get(BASE + '/patient/dashboard', headers=pat_auth, timeout=30)
dump(resp, 'PAT DASH')
resp = requests.get(BASE + '/patient/detection/history', headers=pat_auth, timeout=30)
dump(resp, 'PAT HIST')
resp = requests.get(BASE + '/patient/consultations', headers=pat_auth, timeout=30)
dump(resp, 'PAT CONSULT')
resp = requests.get(BASE + '/doctors', timeout=30)
dump(resp, 'DOCTORS')

resp = requests.post(BASE + '/patient/consultations/request', json={'doctor_id': 1, 'date': '2099-12-31T10:00:00', 'notes': 'Need advice', 'scan_id': None}, headers=pat_auth, timeout=30)
req = dump(resp, 'REQ CREATED')
request_id = req.get('id') if req else None

resp = requests.post(BASE + '/auth/login', json={'email': 'doctor1@example.com', 'password': 'password'}, timeout=30)
doc_login = dump(resp, 'DOC LOGIN')
if not doc_login or 'access_token' not in doc_login:
    raise SystemExit('Doctor login failed')
doc_auth = {'Authorization': f"Bearer {doc_login['access_token']}"}

resp = requests.get(BASE + '/auth/me', headers=doc_auth, timeout=30)
dump(resp, 'DOC ME')
resp = requests.get(BASE + '/doctor/dashboard', headers=doc_auth, timeout=30)
dump(resp, 'DOC DASH')
resp = requests.get(BASE + '/doctor/requests', headers=doc_auth, timeout=30)
dump(resp, 'DOC REQ LIST')

if request_id:
    resp = requests.put(BASE + f'/doctor/requests/{request_id}/status', params={'status': 'accepted'}, headers=doc_auth, timeout=30)
    dump(resp, 'REQ STATUS')
    resp = requests.get(BASE + '/doctor/consultations', headers=doc_auth, timeout=30)
    docs = dump(resp, 'DOC CONSULTS')
    if docs:
        cid = docs[0]['id']
        resp = requests.post(BASE + f'/doctor/consultations/{cid}/notes', json={'notes': 'Reviewed. Follow up in 2 weeks.'}, headers=doc_auth, timeout=30)
        dump(resp, 'NOTES')
else:
    print('NO request id returned')

resp = requests.put(BASE + '/users/profile', json={'name': 'Patient Updated', 'phone': '1234567890', 'bio': 'Test patient'}, headers=pat_auth, timeout=30)
dump(resp, 'PROFILE UPDATED')
