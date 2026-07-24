from app import app

app.testing = True

with app.test_client() as c:
    resp = c.post('/contact', data={'full_name':'Test User','email':'test@example.com','message':'This is a test message.'})
    print('STATUS', resp.status_code)
    data = resp.get_data(as_text=True)
    print('RESPONSE SNIPPET:\n', data[:1000])
