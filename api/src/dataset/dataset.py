import requests
import random
import pandas as pd
import mlflow
import os
import time
from typing import Literal, List
from pydantic import BaseModel
from faker import Faker
import mlflow

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

API_URL = "https://matyschampeyrol-devops-ml.hf.space/api/classify"
EXPERIMENT_NAME = "API_Spam_Classifier_Monitoring"

TIMEOUT_SECONDS = 30 

fake = Faker('en_US')

# --- MODÈLE ---
class Mail(BaseModel):
    to_address: str
    subject: str
    body: str
    label: Literal["SPAM", "HAM"]

# --- DATASET ---
spam_subjects = ["URGENT", "WINNER", "Hot Singles", "Lose Weight", "Bitcoin", "Inheritance"]
ham_subjects = ["Meeting", "Lunch", "Project Update", "Invoice", "Question", "Weekly Report"]

def generate_dataset(n=100) -> List[Mail]:
    dataset = []
    for _ in range(n // 2):
        dataset.append(Mail(to_address=fake.email(), subject=f"{random.choice(spam_subjects)} {random.randint(100,999)}", body=fake.text(), label="SPAM"))
    for _ in range(n // 2):
        dataset.append(Mail(to_address=fake.email(), subject=f"{random.choice(ham_subjects)} - {fake.first_name()}", body=fake.text(), label="HAM"))
    random.shuffle(dataset)
    return dataset

def wakeup_api():
    print("Warm up : cela peut prendre jusqu'à 60s.")
    dummy_mail = {"to_address": "test@test.com", "subject": "Wake up", "body": "test"}
    try:
        requests.post(API_URL, json=dummy_mail, timeout=60)
        print("API réveillée et prête !")
    except Exception as e:
        print(f"Le warm-up a échoué (ou l'API était déjà prête) : {e}")

def run_mlflow_test():
    wakeup_api()

    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        n_mails = 100
        mails = generate_dataset(n_mails)
        print(f"Dataset de {n_mails} mails généré.")
        
        mlflow.log_param("api_url", API_URL)
        mlflow.log_param("dataset_size", n_mails)
        mlflow.log_param("timeout_setting", TIMEOUT_SECONDS)

        correct = 0
        errors_list = []
        timeouts = 0

        print(f"Démarrage des tests API...")
        
        for i, mail in enumerate(mails):
            payload = mail.model_dump(exclude={"label"})
            try:
                response = requests.post(API_URL, json=payload, timeout=TIMEOUT_SECONDS)
                
                if response.status_code == 200:
                    pred = response.json()
                    pred_label = pred.get("label")
                    confidence = pred.get("confidence")

                    if pred_label == mail.label:
                        correct += 1
                    else:
                        errors_list.append({
                            "actual": mail.label,
                            "predicted": pred_label,
                            "confidence": confidence,
                            "subject": mail.subject,
                            "body": mail.body[:100]
                        })
                    
                    if (i+1) % 10 == 0:
                        print(f"   Processed {i+1}/{n_mails}...")

                else:
                    print(f"Erreur HTTP {response.status_code}")

            except requests.exceptions.ReadTimeout:
                print(f"Timeout sur le mail {i+1}")
                timeouts += 1
            except Exception as e:
                print(f"Erreur connexion: {e}")

        accuracy = correct / n_mails if n_mails > 0 else 0
        
        print("-" * 30)
        print(f"Accuracy : {accuracy:.2%}")
        print(f"Timeouts : {timeouts}")
        print("-" * 30)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("timeouts", timeouts)
        mlflow.log_metric("failed_count", len(errors_list))

        if errors_list:
            csv_filename = "classification_errors.csv"
            df_errors = pd.DataFrame(errors_list)
            df_errors.to_csv(csv_filename, index=False)
            
            try:
                print(f"Tentative d'envoi de {csv_filename}...")
                mlflow.log_artifact(csv_filename)
                print("Fichier envoyé avec succès !")
                # On ne supprime le fichier que si l'envoi a réussi
                os.remove(csv_filename)
            except PermissionError:
                print(f"INFO : Impossible d'envoyer l'artefact (Permission Denied).")
                print(f"{os.path.abspath(csv_filename)}")
            except Exception as e:
                print(f"Erreur autre : {e}") 

if __name__ == "__main__":
    run_mlflow_test()