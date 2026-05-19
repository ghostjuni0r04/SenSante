# notebooks/test_groq.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    exit()

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": "Tu es un assistant medical senegalais. Reponds en francais simple. Maximum 3 phrases."},
        {"role": "user",
         "content": "Quels sont les symptomes du paludisme ?"}
    ],
    max_tokens=200,
    temperature=0.3
)

print("=== Reponse de Llama 3 ===")
print(response.choices[0].message.content)
print(f"\nTokens utilises : {response.usage.total_tokens}")

response2 = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system",
         "content": """Tu es un assistant medical senegalais.
Tu recois un diagnostic et des donnees patient.
Explique le resultat en francais simple,
comme un medecin parlerait a son patient.
Sois rassurant mais recommande une consultation.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme."""},
        {"role": "user",
         "content": """Patient : Femme, 28 ans, region Dakar
Symptomes : temperature 39.5, toux, fatigue, maux de tete
Diagnostic du modele : paludisme (probabilite 72%)
Explique ce resultat au patient."""}
    ],
    max_tokens=200,
    temperature=0.3
)
print("=== Explication SenSante ===")
print(response2.choices[0].message.content)

# Exercice 2 : test temperatures
for temp in [0.0, 0.5, 1.0]:
    r = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Tu es un assistant medical senegalais. Reponds en francais simple. Maximum 2 phrases."},
            {"role": "user", "content": "Patient : F, 28 ans, Dakar. Temperature 39.5C. Diagnostic : paludisme (72%). Explique."}
        ],
        max_tokens=150,
        temperature=temp
    )
    print(f"\n=== Temperature {temp} ===")
    print(r.choices[0].message.content)