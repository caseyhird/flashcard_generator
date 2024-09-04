# Flashcard Generator

seenode dash: https://cloud.seenode.com/dashboard/applications?applicationId=965673#logs
seenode URL: https://web-dximtpyqdxz3.up-de-fra1-k8s-1.apps.run-on-seenode.com/

Frontend repo: https://github.com/hathwars/flashcards-frontend
Frontend product URL: https://flashcard-frontend-rosy.vercel.app/

## Local testing

Run the server locally with `python backend/server.py`
You can use the fastapi cli as well, but this is not the way this happens in production

For testing the API, you can use the flashcards script.
To do this and view the generated flashcards, use `python backend/flashcards_script.py --input <input_file> --out <output_file> --log <log_file>`