---
marp: true
theme: gaia
_class: lead
paginate: true
backgroundColor: #f8fafc
color: #0f172a
style: |
  section {
    font-family: 'Inter', sans-serif;
    padding: 40px;
  }
  h1 { color: #6366F1; }
  h2 { color: #312e81; border-bottom: 2px solid #e2e8f0; }
  footer { font-size: 0.5em; color: #64748b; }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 20px;
  }
  .btn {
    background: #6366F1; color: white; padding: 5px 10px; 
    border-radius: 5px; text-decoration: none; font-size: 0.7em;
  }
---

# Architectures Transformer
### Mécanisme d'attention et applications (SST-2)

**Présenté par :** Votre Groupe (2 étudiants)
**Encadreur :** Dr. Stéphane C. K. TEKOUABOU (PhD & Ing.)
**Année Académique :** 2025-2026

---

## 1. Problématique : Les limites des RNN/LSTM

- **Traitement séquentiel :** Les RNN analysent les mots les uns après les autres. Impossible de paralléliser efficacement sur GPU.
- **Oubli à long terme :** Phénomène de disparition du gradient sur les longues phrases.
- **La solution Transformer :** Connecter directement tous les mots entre eux, peu importe leur distance, grâce au mécanisme de *Self-Attention*.

---

## 2. Le Mécanisme de Self-Attention

La formule magique fondamentale de l'architecture :

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Query ($Q$) :** Ce que le mot recherche.
- **Key ($K$) :** Ce que le mot propose aux autres.
- **Value ($V$) :** L'information s'il est sélectionné.
- **$\sqrt{d_k}$ :** Facteur d'échelle pour éviter la saturation des gradients.

---

## 3. Visualisation de l'Attention (SST-2)

<div class="grid">
<div>

- Les mots à forte charge émotionnelle (`wonderful`, `loved`) capturent l'essentiel du poids de l'attention.
- Permet de classifier efficacement la phrase en sentiment **Positif**.

<br>

[👉 Ouvrir le graphique interactif hors-ligne](img/matrice_attention.html){.btn}
</div>
<div>

![w:450 center](img/matrice_attention.png)
</div>
</div>

---

## 4. Encodage Positionnel & Optimisation

<div class="grid">
<div>

- **Position :** Les Transformers n'ayant pas de récurrence, on ajoute des ondes sinusoïdales pour conserver l'ordre des mots.
- **Warmup + Cosine Decay :** Le taux d'apprentissage monte au début pour stabiliser les gradients, puis redescend doucement.

<br>

[👉 Ouvrir la courbe d'optimisation interactive](img/courbe_optimisation.html){.btn}
</div>
<div>

![w:450 center](img/courbe_optimisation.png)
</div>
</div>

---

## 5. Résultats expérimentaux sur SST-2

| Stratégie / Modèle | Paramètres | Accuracy SST-2 |
| :--- | :---: | :---: |
| Régression Logistique (Baseline) | ~0M | 82.3% |
| **Transformer (From Scratch)** | **15.2M** | **87.3%** |
| **DistilBERT (Fine-tuné)** | **66.4M** | **91.5%** |

- Le modèle *From Scratch* surpasse largement les baselines linéaires.
- Le *Fine-tuning* de DistilBERT offre le meilleur compromis précision/vitesse pour un déploiement léger.

---

## Conclusion & Perspectives

- **Succès :** L'architecture Transformer capte parfaitement les dépendances sémantiques complexes sans perte d'information.
- **Futur (Edge AI) :** DistilBERT prouve qu'on peut compresser ces architectures massives pour les déployer sur des architectures à ressources limitées (Sujet J).

Merci pour votre attention ! 
**Place aux questions (Q&A)**.