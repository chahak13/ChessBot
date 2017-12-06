import time
from firebase import firebase
firebase = firebase.FirebaseApplication('https://chessbot-2873e.firebaseio.com/Users/',None)
while True:
    result = firebase.get('move',None)
    print(result)
    time.sleep(1)
