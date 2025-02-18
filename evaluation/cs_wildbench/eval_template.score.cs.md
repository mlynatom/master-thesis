# Instrukce

Jsi odborný hodnotitel. Tvým úkolem je posoudit kvalitu odpovědí generovaných AI modely.
Poskytneme ti uživatelský dotaz a AI-generovanou odpověď.
Nejprve si pečlivě přečti uživatelský dotaz a historii konverzace, abyste správně analyzoval úkol, a poté vyhodnoť kvalitu odpovědi na základě pravidel uvedených níže.

# Konverzace mezi uživatelem a AI

## Historie
<|begin_of_history|>

{$history}

<|end_of_history|> 

## Aktuální uživatelský dotaz
<|begin_of_query|>

{$user_query}

<|end_of_query|>

## Odpověď AI
<|begin_of_response|>

{$model_output}

<|end_of_response|>
 

# Evaluation   

## Kontrolní seznam 

<|begin_of_checklist|>

{$checklist}

<|end_of_checklist|>

Použij tento kontrolní seznam jako vodítko pro své hodnocení, ale neomezuj své posouzení pouze na něj.

## Pravidla 

Vyhodnoť výše uvedenou odpověď na základě analýzy uživatelského dotazu a historie konverzace.
Nejprve sepiš svou analýzu a kontrolní seznam, který jsi použil k hodnocení, a poté poskytni své posouzení podle tohoto seznamu.
Hodnocení je na škále od 1 do 10, kde 1 znamená velmi špatnou odpověď a 10 znamená perfektní odpověď.
Podrobnější kritéria hodnocení jsou následující:

- Hodnocení 1–2: Odpověď je velmi špatná a nedává žádný smysl.
- Hodnocení 3–4: Odpověď je slabá a nepomáhá uživateli vyřešit problém smysluplným způsobem.
- Hodnocení 5–6: Odpověď je průměrná, ale obsahuje problémy (např. faktické chyby, halucinace, chybějící klíčové informace).
- Hodnocení 7–8: Odpověď je dobrá, ale stále existuje prostor pro zlepšení.
- Hodnocení 9–10: Odpověď je vynikající, poskytuje užitečné informace a pomáhá uživateli efektivně vyřešit problém.


## Výstupní formát
Nejprve uveď svou analýzu odpovědi AI a poté shrň své hodnocení ve dvou aspektech: „silné stránky“ a „slabé stránky“. Nakonec urči výsledné skóre.

Výsledky hodnocení uveď v následujícím formátu JSON a nahraď zástupné znaky [] odpovídajícími hodnotami:
```
{
    "strengths": "[analysis for the strengths of the response]",
    "weaknesses": "[analysis for the weaknesses of the response]",
    "score": "[1~10]"
}
```