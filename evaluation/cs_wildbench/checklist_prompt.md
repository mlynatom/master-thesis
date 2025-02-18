# Instrukce
Jsi odborný hodnotitel. Tvým úkolem je pro každou konverzaci napsat 5 otázek, které budou snadno interpretovatelné a jednoduché k ověření.
Tyto otázky budou následně sloužit jako součást kontrolního seznamu k hodnocení kvality odpovědí generovaných AI modely.
Poskytneme ti konverzaci ve formě historie a aktuálního uživatelského dotazu.

#Konverzace

## Historie
<|begin_of_history|>

{history}

<|end_of_history|>

## Aktuální dotaz

<|begin_of_query|>

{user_query}

<|end_of_query|>

# Pravidla pro otázky
Nejprve si sepis analýzu konverzace a následně vytvoř 5 otázek, které by měly být snadno interpretovatelné a jednoduché k ověření.
Otázky budou součástí kontrolního seznamu v promptu pro LLM-as-judge modely, které budou hodnotit jiných výstupy modelů.
Otázky by měly model navést ke správnému hodnocení hodnocení.

## Výstupní formát
Kromě otázek nevracej žádné další informace, ani svou analýzu konverzace.
Každou otázkou uveď jako jednu větu, začínajícím velkým písmenem a končící otazníkem.
Každou otázku odděl novým řádkem.