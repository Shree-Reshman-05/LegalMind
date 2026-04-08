import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import json
import random
from datetime import datetime, date
from functools import wraps

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "legalmind-student-secret-2024")
CORS(app, supports_credentials=True)

# ─────────────────────────────────────────────
# IN-MEMORY DATA STORE (no DB required)
# ─────────────────────────────────────────────
user_progress = {}   # user_id -> progress dict

def get_user(user_id="student"):
    if user_id not in user_progress:
        user_progress[user_id] = {
            "id": user_id,
            "topics_completed": [],
            "quiz_scores": [],
            "cases_attempted": [],
            "cases_correct": 0,
            "streak": 0,
            "last_active": str(date.today()),
            "points": 0,
            "badges": [],
            "level": "Beginner",
            "daily_challenge_done": False,
            "daily_challenge_date": ""
        }
    return user_progress[user_id]

def save_user(user_id, data):
    user_progress[user_id] = data

# ─────────────────────────────────────────────
# CONTENT DATABASE
# ─────────────────────────────────────────────

TOPICS = {
    "contract-law": {
        "id": "contract-law",
        "title": "Contract Law",
        "icon": "📜",
        "level": "Beginner",
        "category": "Civil Law",
        "description": "Master the fundamentals of contract formation, validity, and enforcement.",
        "estimated_time": "25 min",
        "sections": [
            {
                "type": "definition",
                "heading": "What is a Contract?",
                "content": "A contract is a legally binding agreement between two or more parties that creates mutual obligations enforceable by law. It is the cornerstone of commercial and personal legal relationships."
            },
            {
                "type": "elements",
                "heading": "Essential Elements of a Valid Contract",
                "items": [
                    {"label": "Offer", "detail": "A clear proposal made by one party (offeror) to another (offeree), expressing willingness to enter into an agreement on specified terms."},
                    {"label": "Acceptance", "detail": "Unconditional agreement to all terms of the offer. Any change in terms creates a counter-offer, not an acceptance."},
                    {"label": "Consideration", "detail": "Something of value exchanged between parties — money, services, goods, or a promise to act or refrain from acting."},
                    {"label": "Capacity", "detail": "Parties must be legally competent — of legal age (18+), of sound mind, and not disqualified by law."},
                    {"label": "Free Consent", "detail": "Agreement must be free from coercion, undue influence, fraud, misrepresentation, or mistake."},
                    {"label": "Lawful Object", "detail": "The purpose and consideration of the contract must not be illegal, immoral, or opposed to public policy."}
                ]
            },
            {
                "type": "example",
                "heading": "Real-World Example",
                "scenario": "Priya agrees to paint Ravi's house for ₹15,000. Ravi agrees to pay upon completion. Both parties have capacity, there's offer, acceptance, and consideration.",
                "outcome": "This is a valid, enforceable contract."
            },
            {
                "type": "case_reference",
                "heading": "Landmark Case",
                "case": "Carlill v. Carbolic Smoke Ball Co. (1893)",
                "principle": "An advertisement can constitute a valid offer when it is sufficiently definite and shows intention to be bound. Acceptance occurs by performing the conditions stated."
            },
            {
                "type": "flowchart",
                "heading": "Contract Formation Flow",
                "steps": ["Offer Made", "Offer Communicated", "Acceptance Given", "Consideration Exchanged", "Legal Capacity Confirmed", "Free Consent Verified", "Valid Contract ✓"]
            },
            {
                "type": "law_section",
                "heading": "Governing Law",
                "content": "The Indian Contract Act, 1872 governs contracts in India. Section 2(h) defines a contract as 'an agreement enforceable by law'. Section 10 lays down the conditions for valid contracts."
            }
        ]
    },
    "tort-law": {
        "id": "tort-law",
        "title": "Law of Torts",
        "icon": "⚖️",
        "level": "Intermediate",
        "category": "Civil Law",
        "description": "Understand civil wrongs, negligence, and liability in tort law.",
        "estimated_time": "30 min",
        "sections": [
            {
                "type": "definition",
                "heading": "What is a Tort?",
                "content": "A tort is a civil wrong that causes harm or loss to another person, giving the aggrieved party the right to sue for damages. Unlike criminal law, the purpose is to compensate the victim rather than punish the wrongdoer."
            },
            {
                "type": "elements",
                "heading": "Types of Torts",
                "items": [
                    {"label": "Negligence", "detail": "Failure to exercise reasonable care resulting in harm. Requires duty, breach, causation, and damages."},
                    {"label": "Defamation", "detail": "Publication of false statements that damage a person's reputation. Includes libel (written) and slander (spoken)."},
                    {"label": "Trespass", "detail": "Intentional, unlawful entry onto another's land (trespass to land) or interference with their person (trespass to person)."},
                    {"label": "Nuisance", "detail": "Unreasonable interference with a person's use and enjoyment of their land (private nuisance) or public rights (public nuisance)."},
                    {"label": "Strict Liability", "detail": "Liability without fault — applicable when someone brings a dangerous thing onto their land which escapes and causes harm (Rule in Rylands v Fletcher)."}
                ]
            },
            {
                "type": "example",
                "heading": "Negligence Example",
                "scenario": "A doctor performs surgery without adequately warning the patient of known risks. The patient suffers complications that were among those undisclosed risks.",
                "outcome": "The doctor may be liable in tort for medical negligence — breaching the duty of care owed to the patient."
            },
            {
                "type": "case_reference",
                "heading": "Landmark Case",
                "case": "Donoghue v Stevenson (1932)",
                "principle": "Established the modern law of negligence. Lord Atkin's 'neighbour principle' — you must take reasonable care to avoid acts or omissions which you can reasonably foresee would be likely to injure your neighbour."
            },
            {
                "type": "law_section",
                "heading": "Key Principle",
                "content": "India follows English common law principles for tort law. The Motor Vehicles Act, 1988 and Consumer Protection Act, 2019 codify specific tortious liabilities in their domains."
            }
        ]
    },
    "criminal-law": {
        "id": "criminal-law",
        "title": "Criminal Law (IPC)",
        "icon": "🔒",
        "level": "Intermediate",
        "category": "Criminal Law",
        "description": "Study offences, punishments, and defences under the Indian Penal Code.",
        "estimated_time": "35 min",
        "sections": [
            {
                "type": "definition",
                "heading": "What is Criminal Law?",
                "content": "Criminal law defines acts that are offences against society and prescribes punishments. The Indian Penal Code (IPC), 1860 is the primary criminal code in India, supplemented by the Code of Criminal Procedure (CrPC) and Indian Evidence Act."
            },
            {
                "type": "elements",
                "heading": "Elements of a Crime",
                "items": [
                    {"label": "Actus Reus", "detail": "The physical act or conduct that constitutes the criminal offence — the 'guilty act'."},
                    {"label": "Mens Rea", "detail": "The mental element — criminal intent, knowledge, or recklessness. Most crimes require both actus reus and mens rea."},
                    {"label": "Causation", "detail": "A direct causal link between the defendant's act and the prohibited result."},
                    {"label": "Concurrence", "detail": "The mental state and physical act must exist simultaneously."}
                ]
            },
            {
                "type": "example",
                "heading": "Murder vs Culpable Homicide",
                "scenario": "A shoots B intending to kill. B dies. Compare with: X gives a light push to Y (elderly), Y falls and dies — X did not intend death.",
                "outcome": "First case: Murder under IPC Section 302. Second case: Culpable Homicide Not Amounting to Murder — lesser culpability, lesser punishment."
            },
            {
                "type": "case_reference",
                "heading": "Landmark Case",
                "case": "K.M. Nanavati v. State of Maharashtra (1962)",
                "principle": "Established that provocation must be sudden and grave for the defence to reduce murder to culpable homicide. Tested the limits of the 'grave and sudden provocation' exception."
            },
            {
                "type": "law_section",
                "heading": "Key IPC Sections",
                "content": "Section 299: Culpable Homicide | Section 300: Murder | Section 302: Punishment for Murder (death or life imprisonment) | Section 304: Punishment for Culpable Homicide | Section 378: Theft | Section 415: Cheating"
            }
        ]
    },
    "constitutional-law": {
        "id": "constitutional-law",
        "title": "Constitutional Law",
        "icon": "🏛️",
        "level": "Advanced",
        "category": "Public Law",
        "description": "Explore fundamental rights, directive principles, and constitutional interpretation.",
        "estimated_time": "40 min",
        "sections": [
            {
                "type": "definition",
                "heading": "The Constitution of India",
                "content": "The Constitution of India, adopted on 26 November 1949, is the supreme law of India. It establishes the framework of government, fundamental rights and duties of citizens, and directive principles for state policy."
            },
            {
                "type": "elements",
                "heading": "Fundamental Rights (Part III)",
                "items": [
                    {"label": "Article 14", "detail": "Right to Equality — Equality before law and equal protection of laws."},
                    {"label": "Article 19", "detail": "Freedom of Speech, Expression, Assembly, Movement, Residence, and Profession (with reasonable restrictions)."},
                    {"label": "Article 21", "detail": "Right to Life and Personal Liberty — No person shall be deprived of their life or personal liberty except according to procedure established by law. Most expansively interpreted right."},
                    {"label": "Article 32", "detail": "Right to Constitutional Remedies — Dr. Ambedkar called this the 'heart and soul' of the Constitution. Allows citizens to approach the Supreme Court directly."},
                    {"label": "Article 25-28", "detail": "Right to Freedom of Religion — Freedom of conscience and free profession, practice and propagation of religion."}
                ]
            },
            {
                "type": "example",
                "heading": "Article 21 in Action",
                "scenario": "The state detains a person without trial for years. The person files a writ petition claiming violation of Article 21.",
                "outcome": "Supreme Court can direct immediate release. Article 21 has been interpreted to include right to dignity, education, health, privacy, and livelihood."
            },
            {
                "type": "case_reference",
                "heading": "Landmark Case",
                "case": "Kesavananda Bharati v. State of Kerala (1973)",
                "principle": "Established the 'Basic Structure Doctrine' — Parliament can amend the Constitution but cannot alter its basic structure (democracy, federalism, separation of powers, fundamental rights). The most important constitutional case in Indian history."
            },
            {
                "type": "law_section",
                "heading": "Writs Available",
                "content": "Habeas Corpus: Produce the body (liberty) | Mandamus: Command to perform duty | Prohibition: Prohibit inferior court from exceeding jurisdiction | Certiorari: Quash inferior court order | Quo Warranto: By what authority (public office)"
            }
        ]
    },
    "property-law": {
        "id": "property-law",
        "title": "Property Law",
        "icon": "🏠",
        "level": "Advanced",
        "category": "Civil Law",
        "description": "Navigate the law of immovable property, transfers, and ownership rights.",
        "estimated_time": "30 min",
        "sections": [
            {
                "type": "definition",
                "heading": "What is Property Law?",
                "content": "Property law governs the various forms of ownership and tenancy in real property (land and buildings) and personal property. The Transfer of Property Act, 1882 is the primary statute governing transfer of immovable property in India."
            },
            {
                "type": "elements",
                "heading": "Modes of Transfer",
                "items": [
                    {"label": "Sale", "detail": "Transfer of ownership in exchange for a price paid or promised. Under TPA Section 54."},
                    {"label": "Mortgage", "detail": "Transfer of interest in specific immovable property to secure repayment of money. Six types including simple mortgage and English mortgage."},
                    {"label": "Lease", "detail": "Transfer of right to enjoy property for a defined period for consideration. Governed by TPA Sections 105-117."},
                    {"label": "Gift", "detail": "Voluntary transfer of property without consideration. Must be accepted by donee during donor's lifetime."},
                    {"label": "Exchange", "detail": "Transfer of ownership of one thing for ownership of another thing — essentially a barter."}
                ]
            },
            {
                "type": "example",
                "heading": "Property Dispute Example",
                "scenario": "A sells his house to B for ₹50 lakhs. A registered sale deed is executed. Later, A's son claims the house was his share in ancestral property.",
                "outcome": "If the property was self-acquired by A (not ancestral), the sale to B is valid. If ancestral, the son may have a share claim under Hindu Succession Act."
            },
            {
                "type": "case_reference",
                "heading": "Landmark Case",
                "case": "Vineeta Sharma v. Rakesh Sharma (2020)",
                "principle": "Supreme Court held that daughters have equal coparcenary rights in Hindu Undivided Family property from birth, regardless of whether the father was alive when the Hindu Succession (Amendment) Act, 2005 came into force."
            },
            {
                "type": "law_section",
                "heading": "Key Statutes",
                "content": "Transfer of Property Act, 1882 | Registration Act, 1908 (documents requiring registration) | Hindu Succession Act, 1956 (as amended in 2005) | Indian Stamp Act, 1899 (stamp duty on instruments)"
            }
        ]
    }
}

CASE_STUDIES = [
    {
        "id": "cs-001",
        "title": "The Broken Promise",
        "topic": "Contract Law",
        "difficulty": "Beginner",
        "scenario": """Ram agrees to sell his motorcycle to Shyam for ₹45,000. They verbally agree that Shyam will pay within 7 days and Ram will deliver the vehicle. On the 5th day, Ram sells the motorcycle to someone else for ₹50,000, claiming the verbal agreement was not binding.""",
        "question": "Is Ram liable for breach of contract? Does Shyam have any legal remedy?",
        "options": [
            "No. Verbal contracts have no legal validity in India.",
            "Yes. A verbal contract is valid; Ram is liable and Shyam can seek damages.",
            "It depends — only if there were witnesses to the verbal agreement.",
            "No. Ram was justified since he got a better price."
        ],
        "correct": 1,
        "judgment": "Ram is indeed liable for breach of contract. Under the Indian Contract Act, 1872, oral contracts are generally valid and enforceable (with exceptions for certain types like sale of immovable property which require written agreements). The contract was complete — there was offer (Ram's agreement to sell), acceptance (Shyam's agreement to buy), and consideration (₹45,000). Shyam can sue for specific performance or damages.",
        "law_section": "Indian Contract Act §10, §73 (Compensation for breach), Sale of Goods Act, 1930",
        "principle": "Verbal contracts are legally valid in India for movable property. Breach of contract entitles the aggrieved party to damages or specific performance."
    },
    {
        "id": "cs-002",
        "title": "The Negligent Doctor",
        "topic": "Law of Torts",
        "difficulty": "Intermediate",
        "scenario": """Dr. Kapoor performs a routine appendectomy on Meena. He fails to inform her that there is a 3% risk of nerve damage — a known risk he should have disclosed. Meena suffers nerve damage and is partially paralyzed. She was otherwise healthy before the surgery.""",
        "question": "Can Meena successfully sue Dr. Kapoor for medical negligence?",
        "options": [
            "No. Surgery always carries risks and doctors cannot guarantee outcomes.",
            "No. A 3% risk is too small to require mandatory disclosure.",
            "Yes. Failure to obtain informed consent constitutes medical negligence.",
            "Only if she can prove the doctor performed the surgery incorrectly."
        ],
        "correct": 2,
        "judgment": "Meena has a valid claim. Medical negligence in India encompasses failure to obtain 'informed consent' — the duty to disclose known, material risks. The Supreme Court in Jacob Mathew v. State of Punjab (2005) established that a doctor owes patients a duty of care. The doctor's failure to disclose the risk (even at 3%) prevented Meena from making an informed decision. This constitutes breach of duty, and causation is established.",
        "law_section": "Consumer Protection Act, 2019 | Indian Medical Council (Professional Conduct) Regulations, 2002",
        "principle": "Informed consent is a fundamental right of patients. Failure to disclose known material risks can constitute medical negligence even if the surgery itself was performed correctly."
    },
    {
        "id": "cs-003",
        "title": "The Unlawful Detention",
        "topic": "Constitutional Law",
        "difficulty": "Intermediate",
        "scenario": """Arjun, a student activist, is arrested by police during a peaceful protest. He is held for 72 hours without being produced before a magistrate. His family is not informed. When his lawyer tries to meet him, police refuse access. No FIR has been registered.""",
        "question": "Which of Arjun's constitutional rights have been violated and what remedy is available?",
        "options": [
            "No rights violated — police have broad powers during protests.",
            "Only his right to counsel is violated.",
            "Article 21 (right to liberty) and Article 22 (protection against arrest) are violated; he can file for Habeas Corpus.",
            "He must first file a complaint with the police commissioner."
        ],
        "correct": 2,
        "judgment": "Multiple fundamental rights have been violated. Article 22(1) mandates that an arrested person must be informed of grounds of arrest, and cannot be denied legal counsel. Article 22(2) requires production before a magistrate within 24 hours. Article 21 protects personal liberty from arbitrary state action. Arjun or his family can file a writ of Habeas Corpus under Article 32 before the Supreme Court or Article 226 before the High Court for immediate release.",
        "law_section": "Constitution of India — Articles 21, 22, 32 | CrPC Section 57 (24-hour rule)",
        "principle": "Detention without magistrate production within 24 hours is unconstitutional. Habeas Corpus is the primary remedy against unlawful detention."
    },
    {
        "id": "cs-004",
        "title": "The Forged Property",
        "topic": "Property Law",
        "difficulty": "Advanced",
        "scenario": """Suresh claims to own a plot of land and sells it to Rajesh for ₹30 lakhs. They execute a registered sale deed. Later, it emerges that Suresh had forged documents — the actual owner is Geeta, who never sold the property. Rajesh paid in good faith without knowledge of fraud.""",
        "question": "What happens to Rajesh's purchase? Can Geeta recover her property?",
        "options": [
            "Rajesh keeps the property since the sale deed was registered.",
            "Geeta recovers the property; Rajesh can only sue Suresh for the money.",
            "The property is split equally between Geeta and Rajesh.",
            "Government takes over the property as it was obtained fraudulently."
        ],
        "correct": 1,
        "judgment": "Geeta can recover her property despite the registered sale deed. The principle is 'nemo dat quod non habet' — no one can give what they do not have. Suresh had no title to convey. Registration does not validate an otherwise fraudulent transfer. Rajesh, as a bona fide purchaser, can sue Suresh for recovery of the purchase price plus damages. Geeta may also be entitled to mesne profits for the period Rajesh was in possession.",
        "law_section": "Transfer of Property Act §53 (Fraudulent transfer) | Indian Penal Code §420 (Cheating) | Registration Act, 1908",
        "principle": "Fraud vitiates title. A seller cannot transfer better title than they possess. Registration of a forged document does not make it valid."
    },
    {
        "id": "cs-005",
        "title": "The Social Media Post",
        "topic": "Law of Torts",
        "difficulty": "Beginner",
        "scenario": """Priya posts on Instagram: 'Avoid XYZ Restaurant — they served me rotten food and gave me food poisoning. Worst dining experience ever!' The restaurant owner threatens to sue for defamation. Priya's claims are true and she has a medical certificate showing food poisoning.""",
        "question": "Can the restaurant successfully sue Priya for defamation?",
        "options": [
            "Yes. Any negative review that damages business reputation is defamation.",
            "No. Truth is a complete defence to defamation; Priya is protected.",
            "Yes, because the post was published to thousands of people.",
            "It depends on whether the restaurant is a private or public company."
        ],
        "correct": 1,
        "judgment": "The restaurant cannot successfully sue Priya for defamation. Under Indian law (Section 499 IPC and tort law), truth is an absolute defence to defamation. Priya's statement was true, supported by medical evidence. Additionally, fair comment on matters of public interest (like consumer experiences) is a valid defence. Publishing a genuine negative review based on actual experience is protected speech. False reviews would be defamatory, but truthful ones are protected.",
        "law_section": "IPC Section 499 (Defamation) — Exception 1: Truth for Public Good | Consumer Protection Act, 2019",
        "principle": "Truth is a complete defence to defamation. Honest, truthful consumer reviews are protected under freedom of speech and consumer rights."
    },
    {
        "id": "cs-006",
        "title": "The Midnight Escape",
        "topic": "Criminal Law",
        "difficulty": "Advanced",
        "scenario": """Vikram breaks into an empty house at night with the intention to steal valuables. Inside, he discovers a person hiding (the owner who had returned unexpectedly). Surprised, Vikram pushes the owner aside to flee. The owner falls, hits his head, and dies. Vikram claims he only intended to steal, not to kill.""",
        "question": "What offence is Vikram most likely guilty of?",
        "options": [
            "Only theft, since he did not intend to kill anyone.",
            "Murder under Section 302 IPC.",
            "Culpable homicide and robbery, as death occurred during commission of dacoity/robbery.",
            "Accidental death — since he didn't intend it, no criminal liability."
        ],
        "correct": 2,
        "judgment": "Vikram is guilty of multiple offences. The housebreaking constitutes burglary (IPC §457). The use of force during the commission of theft elevates it to robbery (IPC §390). The death occurring during the robbery makes this a case of culpable homicide amounting to murder under IPC §300, clause 4 — an act done with the intention of causing bodily injury to any person and the act is so imminently dangerous that it must in all probability cause death. The concept of 'constructive murder' applies here.",
        "law_section": "IPC §300 (Murder, clause 4), §302 (Punishment for murder), §390 (Robbery), §457 (Lurking house-trespass)",
        "principle": "A death resulting from the use of force during the commission of robbery constitutes murder, regardless of the offender's primary intention being theft."
    }
]

QUIZ_QUESTIONS = [
    {
        "id": "q001",
        "topic": "Contract Law",
        "difficulty": "Beginner",
        "type": "MCQ",
        "question": "Which Section of the Indian Contract Act defines a 'Contract'?",
        "options": ["Section 2(e)", "Section 2(h)", "Section 10", "Section 73"],
        "correct": 1,
        "explanation": "Section 2(h) of the Indian Contract Act, 1872 defines a contract as 'an agreement enforceable by law'. Section 2(e) defines 'agreement', Section 10 gives conditions for valid contracts, and Section 73 deals with compensation for breach."
    },
    {
        "id": "q002",
        "topic": "Contract Law",
        "difficulty": "Beginner",
        "type": "True/False",
        "question": "A minor can enter into a valid contract in India.",
        "options": ["True", "False"],
        "correct": 1,
        "explanation": "FALSE. Under Section 11 of the Indian Contract Act, a minor (under 18 years) lacks contractual capacity. Contracts entered into by minors are void ab initio (void from the beginning), not merely voidable."
    },
    {
        "id": "q003",
        "topic": "Criminal Law",
        "difficulty": "Beginner",
        "type": "MCQ",
        "question": "Under which section of IPC is 'Murder' defined?",
        "options": ["Section 299", "Section 300", "Section 302", "Section 304"],
        "correct": 1,
        "explanation": "Section 300 IPC defines 'Murder'. Section 299 defines 'Culpable Homicide', Section 302 provides the punishment for murder (death or life imprisonment + fine), and Section 304 provides punishment for culpable homicide not amounting to murder."
    },
    {
        "id": "q004",
        "topic": "Constitutional Law",
        "difficulty": "Intermediate",
        "type": "MCQ",
        "question": "Which Article of the Constitution guarantees the Right to Life and Personal Liberty?",
        "options": ["Article 14", "Article 19", "Article 21", "Article 32"],
        "correct": 2,
        "explanation": "Article 21 guarantees that 'No person shall be deprived of his life or personal liberty except according to procedure established by law.' It is the most litigated fundamental right and has been interpreted to include rights to dignity, privacy, health, education, and more."
    },
    {
        "id": "q005",
        "topic": "Constitutional Law",
        "difficulty": "Intermediate",
        "type": "Scenario",
        "question": "A state law requires all vehicles to display religious symbols. A citizen challenges this law. Which Article would primarily be invoked?",
        "options": [
            "Article 14 — Equality before law",
            "Article 25 — Freedom of religion",
            "Article 19(1)(a) — Freedom of speech",
            "Article 32 — Right to constitutional remedies"
        ],
        "correct": 1,
        "explanation": "Article 25 guarantees freedom of conscience and free profession, practice and propagation of religion. Compelling citizens to display religious symbols violates this right as it infringes on freedom of conscience. Article 14 (equality) could also apply as it discriminates, and Article 32 is the remedy to approach the Supreme Court."
    },
    {
        "id": "q006",
        "topic": "Law of Torts",
        "difficulty": "Intermediate",
        "type": "MCQ",
        "question": "The landmark case that established the modern law of negligence and the 'neighbour principle' was:",
        "options": [
            "Rylands v Fletcher (1868)",
            "Donoghue v Stevenson (1932)",
            "Caparo Industries v Dickman (1990)",
            "Blyth v Birmingham Waterworks (1856)"
        ],
        "correct": 1,
        "explanation": "Donoghue v Stevenson (1932) is the foundational case for negligence law. Lord Atkin's 'neighbour principle' stated that you must take reasonable care to avoid acts or omissions which you can reasonably foresee would be likely to injure your neighbour — i.e., those persons so closely and directly affected by your act that you ought reasonably to have them in contemplation."
    },
    {
        "id": "q007",
        "topic": "Property Law",
        "difficulty": "Advanced",
        "type": "True/False",
        "question": "A daughter has equal coparcenary rights in Hindu Undivided Family property from birth, regardless of whether her father was alive in 2005.",
        "options": ["True", "False"],
        "correct": 0,
        "explanation": "TRUE. The Supreme Court in Vineeta Sharma v. Rakesh Sharma (2020) held that daughters have equal coparcenary rights from birth under the Hindu Succession (Amendment) Act, 2005, regardless of whether the father was alive on September 9, 2005 (when the amendment came into force). This overruled earlier conflicting judgments."
    },
    {
        "id": "q008",
        "topic": "Contract Law",
        "difficulty": "Intermediate",
        "type": "Scenario",
        "question": "A promises to pay B ₹10,000 if B stops his daily jogging. B stops jogging for a month. Is this a valid contract?",
        "options": [
            "Yes — stopping an activity is valid consideration",
            "No — consideration must always be monetary",
            "No — the contract lacks a lawful object",
            "Only if B was a professional runner"
        ],
        "correct": 0,
        "explanation": "Yes, this is a valid contract with valid consideration. Under the Indian Contract Act, consideration can be an act, abstinence, or promise. B's abstinence from jogging (something he has a right to do) constitutes valid consideration. The principle: 'A valuable consideration may consist in some right, interest, profit or benefit accruing to one party, or some forbearance, detriment, loss or responsibility suffered or undertaken by the other.'"
    },
    {
        "id": "q009",
        "topic": "Criminal Law",
        "difficulty": "Advanced",
        "type": "Scenario",
        "question": "A, in grave and sudden provocation, shoots at B but misses and hits C instead. C dies. A is charged with murder of C. What is the likely verdict?",
        "options": [
            "A is guilty of murder — provocation is not a defence for killing an unintended victim",
            "A is guilty of culpable homicide — provocation reduces the offence",
            "A is not guilty — C's death was accidental",
            "A is guilty of grievous hurt only"
        ],
        "correct": 1,
        "explanation": "Under IPC Section 301 (Culpable homicide by causing death of person other than person whose death was intended), the doctrine of 'transferred malice' applies — A's guilty intent transfers to C. However, the grave and sudden provocation exception under Exception 1 to Section 300 can reduce the offence to culpable homicide not amounting to murder (IPC §304), if the provocation was genuinely sudden and grave. The exact verdict depends on facts."
    },
    {
        "id": "q010",
        "topic": "Constitutional Law",
        "difficulty": "Advanced",
        "type": "MCQ",
        "question": "The 'Basic Structure Doctrine' — which holds Parliament cannot alter the fundamental features of the Constitution — was established in:",
        "options": [
            "A.K. Gopalan v. State of Madras (1950)",
            "Golaknath v. State of Punjab (1967)",
            "Kesavananda Bharati v. State of Kerala (1973)",
            "Minerva Mills v. Union of India (1980)"
        ],
        "correct": 2,
        "explanation": "Kesavananda Bharati v. State of Kerala (1973) is the most important constitutional case in Indian history. A 13-judge bench (the largest ever) held by 7:6 majority that while Parliament can amend any part of the Constitution under Article 368, it cannot alter its 'basic structure'. This includes parliamentary democracy, federalism, separation of powers, judicial review, and fundamental rights."
    },
    {
        "id": "q011",
        "topic": "Law of Torts",
        "difficulty": "Beginner",
        "type": "True/False",
        "question": "Under the rule in Rylands v Fletcher, liability is imposed only if the defendant was negligent.",
        "options": ["True", "False"],
        "correct": 1,
        "explanation": "FALSE. The rule in Rylands v Fletcher (1868) imposes STRICT LIABILITY — liability without fault. If a person accumulates something on their land that is likely to do mischief if it escapes, and it does escape and causes damage, the person is liable even without negligence. The key elements are: non-natural use of land, accumulation of something dangerous, escape, and resulting damage."
    },
    {
        "id": "q012",
        "topic": "Property Law",
        "difficulty": "Beginner",
        "type": "MCQ",
        "question": "Which Act primarily governs the transfer of immovable property in India?",
        "options": [
            "Registration Act, 1908",
            "Indian Stamp Act, 1899",
            "Transfer of Property Act, 1882",
            "Real Estate Act, 2016"
        ],
        "correct": 2,
        "explanation": "The Transfer of Property Act (TPA), 1882 is the primary statute governing transfers of immovable property in India. It defines concepts like sale, mortgage, lease, gift, and exchange of property. The Registration Act governs the registration requirement for documents, and the Stamp Act governs stamp duty payable on instruments."
    }
]

DAILY_CHALLENGES = [
    {
        "id": "dc-001",
        "title": "The Midnight Promise",
        "time_limit": 300,
        "type": "case",
        "scenario": "At 2 AM, drunk Ram texts Shyam: 'I'll sell you my bike for ₹100'. Shyam immediately replies 'Deal!' and transfers ₹100. Ram, sober the next morning, refuses to complete the sale.",
        "question": "Is there a valid, enforceable contract?",
        "options": [
            "Yes — the contract was formed at 2 AM and is binding",
            "No — contracts made while drunk are void as there was no free consent",
            "Yes — but only if Shyam can prove Ram was actually drunk",
            "No — ₹100 is inadequate consideration for a bike"
        ],
        "correct": 1,
        "explanation": "A contract made under the influence of alcohol may be voidable at the option of the intoxicated party (Ram). Under IPC, a person is of 'unsound mind' if they are incapable of understanding the contract at the time. If Ram was so drunk he couldn't understand what he was doing, the contract lacks free consent and is voidable. Shyam gets his ₹100 back. Note: inadequacy of consideration alone is not ground for invalidity — courts don't assess adequacy, only presence of consideration.",
        "points": 50,
        "badge": "Night Owl Jurist"
    },
    {
        "id": "dc-002",
        "title": "The Viral Video",
        "time_limit": 300,
        "type": "case",
        "scenario": "Nita secretly records her neighbor Kiran having a private conversation and posts it on YouTube without consent. The video gets 2 million views. Kiran is humiliated and suffers mental distress.",
        "question": "What legal remedies does Kiran have?",
        "options": [
            "None — recording conversations in your own home is legal",
            "Only a moral complaint — there's no specific law against this",
            "Kiran can claim under privacy rights (Article 21), IT Act, and potentially defamation/nuisance torts",
            "Kiran can only sue YouTube, not Nita"
        ],
        "correct": 2,
        "explanation": "Kiran has multiple legal remedies. The Supreme Court in K.S. Puttaswamy v. Union of India (2017) recognized privacy as a fundamental right under Article 21. The IT Act, 2000 (Sections 66E — violation of privacy, Section 67 — publishing obscene/private content) provides criminal remedies. Additionally, Kiran can sue in tort for invasion of privacy, intentional infliction of emotional distress, and potentially defamation if the content damaged her reputation. Injunction + damages can be claimed.",
        "points": 50,
        "badge": "Digital Rights Defender"
    }
]

BADGES_DATA = {
    "First Step": "Completed your first topic",
    "Quiz Master": "Scored 100% in a quiz",
    "Case Solver": "Solved 5 case studies",
    "Streak Week": "Maintained a 7-day streak",
    "Night Owl Jurist": "Won a daily challenge",
    "Digital Rights Defender": "Won a daily challenge",
    "Legal Eagle": "Reached Advanced level",
    "Contract Champion": "Mastered Contract Law",
    "Perfect Score": "Achieved 100% quiz accuracy"
}

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("student_index.html")

@app.route("/api/topics", methods=["GET"])
def get_topics():
    user = get_user()
    topics_list = []
    for tid, topic in TOPICS.items():
        t = {
            "id": topic["id"],
            "title": topic["title"],
            "icon": topic["icon"],
            "level": topic["level"],
            "category": topic["category"],
            "description": topic["description"],
            "estimated_time": topic["estimated_time"],
            "completed": topic["id"] in user["topics_completed"]
        }
        topics_list.append(t)
    return jsonify({"topics": topics_list})

@app.route("/api/topic/<topic_id>", methods=["GET"])
def get_topic(topic_id):
    if topic_id not in TOPICS:
        return jsonify({"error": "Topic not found"}), 404
    return jsonify({"topic": TOPICS[topic_id]})

@app.route("/api/topic/<topic_id>/complete", methods=["POST"])
def complete_topic(topic_id):
    user = get_user()
    if topic_id not in user["topics_completed"]:
        user["topics_completed"].append(topic_id)
        user["points"] += 20
        if len(user["topics_completed"]) == 1 and "First Step" not in user["badges"]:
            user["badges"].append("First Step")
        _update_level(user)
    save_user("student", user)
    return jsonify({"success": True, "points": user["points"], "badges": user["badges"]})

@app.route("/api/cases", methods=["GET"])
def get_cases():
    user = get_user()
    cases = []
    for case in CASE_STUDIES:
        c = {
            "id": case["id"],
            "title": case["title"],
            "topic": case["topic"],
            "difficulty": case["difficulty"],
            "attempted": case["id"] in user["cases_attempted"]
        }
        cases.append(c)
    return jsonify({"cases": cases})

@app.route("/api/case/<case_id>", methods=["GET"])
def get_case(case_id):
    case = next((c for c in CASE_STUDIES if c["id"] == case_id), None)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    # return without answer
    c = {k: v for k, v in case.items() if k != "correct"}
    return jsonify({"case": c})

@app.route("/api/case/<case_id>/submit", methods=["POST"])
def submit_case(case_id):
    data = request.get_json()
    answer = data.get("answer", -1)
    case = next((c for c in CASE_STUDIES if c["id"] == case_id), None)
    if not case:
        return jsonify({"error": "Case not found"}), 404
    
    user = get_user()
    correct = (answer == case["correct"])
    
    if case_id not in user["cases_attempted"]:
        user["cases_attempted"].append(case_id)
        if correct:
            user["cases_correct"] += 1
            user["points"] += 30
        if len(user["cases_attempted"]) >= 5 and "Case Solver" not in user["badges"]:
            user["badges"].append("Case Solver")
    
    save_user("student", user)
    return jsonify({
        "correct": correct,
        "correct_answer": case["correct"],
        "judgment": case["judgment"],
        "law_section": case["law_section"],
        "principle": case["principle"]
    })

@app.route("/api/quiz", methods=["GET"])
def get_quiz():
    topic = request.args.get("topic", None)
    difficulty = request.args.get("difficulty", None)
    count = int(request.args.get("count", 5))
    
    questions = QUIZ_QUESTIONS
    if topic:
        questions = [q for q in questions if q["topic"] == topic]
    if difficulty:
        questions = [q for q in questions if q["difficulty"] == difficulty]
    
    selected = random.sample(questions, min(count, len(questions)))
    # strip correct answer
    safe = [{k: v for k, v in q.items() if k != "correct"} for q in selected]
    return jsonify({"questions": safe, "total": len(selected)})

@app.route("/api/quiz/submit", methods=["POST"])
def submit_quiz():
    data = request.get_json()
    answers = data.get("answers", {})  # {question_id: selected_index}
    
    results = []
    score = 0
    for q in QUIZ_QUESTIONS:
        if q["id"] in answers:
            user_ans = answers[q["id"]]
            correct = (user_ans == q["correct"])
            if correct:
                score += 1
            results.append({
                "id": q["id"],
                "correct": correct,
                "user_answer": user_ans,
                "correct_answer": q["correct"],
                "explanation": q["explanation"]
            })
    
    total = len(results)
    pct = round((score / total) * 100) if total > 0 else 0
    
    user = get_user()
    user["quiz_scores"].append({"score": pct, "date": str(date.today()), "total": total, "correct": score})
    user["points"] += score * 10
    
    if pct == 100 and "Quiz Master" not in user["badges"]:
        user["badges"].append("Quiz Master")
    if pct == 100 and "Perfect Score" not in user["badges"]:
        user["badges"].append("Perfect Score")
    
    _update_level(user)
    save_user("student", user)
    
    return jsonify({
        "score": score,
        "total": total,
        "percentage": pct,
        "results": results,
        "points_earned": score * 10
    })

@app.route("/api/daily-challenge", methods=["GET"])
def get_daily_challenge():
    user = get_user()
    today = str(date.today())
    done = user.get("daily_challenge_date") == today
    challenge = random.choice(DAILY_CHALLENGES)
    c = {k: v for k, v in challenge.items() if k != "correct"}
    return jsonify({"challenge": c, "already_done": done})

@app.route("/api/daily-challenge/submit", methods=["POST"])
def submit_daily_challenge():
    data = request.get_json()
    challenge_id = data.get("challenge_id")
    answer = data.get("answer", -1)
    
    challenge = next((c for c in DAILY_CHALLENGES if c["id"] == challenge_id), None)
    if not challenge:
        return jsonify({"error": "Challenge not found"}), 404
    
    user = get_user()
    today = str(date.today())
    correct = (answer == challenge["correct"])
    
    if correct and user.get("daily_challenge_date") != today:
        user["points"] += challenge["points"]
        user["daily_challenge_date"] = today
        # streak
        yesterday = str(date.fromordinal(date.today().toordinal() - 1))
        if user.get("last_active") == yesterday:
            user["streak"] += 1
        else:
            user["streak"] = 1
        user["last_active"] = today
        
        badge = challenge.get("badge")
        if badge and badge not in user["badges"]:
            user["badges"].append(badge)
        if user["streak"] >= 7 and "Streak Week" not in user["badges"]:
            user["badges"].append("Streak Week")
    
    save_user("student", user)
    return jsonify({
        "correct": correct,
        "correct_answer": challenge["correct"],
        "explanation": challenge["explanation"],
        "points_earned": challenge["points"] if correct else 0,
        "streak": user["streak"]
    })

@app.route("/api/progress", methods=["GET"])
def get_progress():
    user = get_user()
    total_topics = len(TOPICS)
    completed = len(user["topics_completed"])
    cases_done = len(user["cases_attempted"])
    total_cases = len(CASE_STUDIES)
    case_accuracy = round((user["cases_correct"] / cases_done * 100)) if cases_done > 0 else 0
    
    scores = [s["score"] for s in user["quiz_scores"]]
    avg_quiz = round(sum(scores) / len(scores)) if scores else 0
    
    return jsonify({
        "topics_completed": completed,
        "total_topics": total_topics,
        "topics_percent": round((completed / total_topics) * 100) if total_topics else 0,
        "cases_attempted": cases_done,
        "total_cases": total_cases,
        "case_accuracy": case_accuracy,
        "quiz_scores": user["quiz_scores"][-5:],
        "avg_quiz_score": avg_quiz,
        "streak": user["streak"],
        "points": user["points"],
        "level": user["level"],
        "badges": user["badges"]
    })

@app.route("/api/leaderboard", methods=["GET"])
def get_leaderboard():
    # Simulated leaderboard
    board = [
        {"rank": 1, "name": "Arjun M.", "points": 1450, "level": "Advanced", "streak": 12},
        {"rank": 2, "name": "Priya K.", "points": 1200, "level": "Advanced", "streak": 8},
        {"rank": 3, "name": "You", "points": get_user()["points"], "level": get_user()["level"], "streak": get_user()["streak"]},
        {"rank": 4, "name": "Ravi S.", "points": 890, "level": "Intermediate", "streak": 5},
        {"rank": 5, "name": "Meena R.", "points": 720, "level": "Intermediate", "streak": 3},
    ]
    board.sort(key=lambda x: x["points"], reverse=True)
    for i, entry in enumerate(board):
        entry["rank"] = i + 1
    return jsonify({"leaderboard": board})

def _update_level(user):
    pts = user["points"]
    completed = len(user["topics_completed"])
    if pts >= 500 or completed >= 4:
        user["level"] = "Advanced"
        if "Legal Eagle" not in user["badges"]:
            user["badges"].append("Legal Eagle")
    elif pts >= 200 or completed >= 2:
        user["level"] = "Intermediate"
    else:
        user["level"] = "Beginner"

if __name__ == "__main__":
    print("\n" + "="*60)
    print("⚖️  LegalMind Student Mode")
    print("="*60)
    print("📍  http://localhost:5000")
    print(f"📚  {len(TOPICS)} Topics | {len(CASE_STUDIES)} Cases | {len(QUIZ_QUESTIONS)} Quiz Questions")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)