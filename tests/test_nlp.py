"""
Unit tests for the NLP Engine components.

Run with:
    pytest tests/test_nlp.py -v
"""
import pytest


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestNormalizer:
    def test_basic_normalization(self):
        from nlp_engine.preprocessing.normalizer import normalize
        assert normalize("  Hello  WORLD!  ") == "hello world!"

    def test_remove_punctuation(self):
        from nlp_engine.preprocessing.normalizer import normalize
        result = normalize("What are the fees??", remove_punctuation=True)
        assert "?" not in result
        assert "fees" in result

    def test_remove_digits(self):
        from nlp_engine.preprocessing.normalizer import normalize
        result = normalize("CS 401 course", remove_digits=True)
        assert "401" not in result

    def test_empty_string(self):
        from nlp_engine.preprocessing.normalizer import normalize
        assert normalize("") == ""

    def test_whitespace_collapse(self):
        from nlp_engine.preprocessing.normalizer import normalize
        assert normalize("hello   world") == "hello world"

    def test_normalize_for_classification(self):
        from nlp_engine.preprocessing.normalizer import normalize_for_classification
        result = normalize_for_classification("What are the CS fees??")
        assert result == result.lower()
        assert "?" not in result

    def test_unicode_normalization(self):
        from nlp_engine.preprocessing.normalizer import normalize
        # Accented characters should survive NFC normalization
        result = normalize("café")
        assert result == "café"

    def test_arabic_normalization_basic(self):
        from nlp_engine.preprocessing.normalizer import normalize
        text = "السَّلامُ عليكم يا أحمد في عام ٢٠٢٤"
        result = normalize(text)
        assert "السلام" in result
        assert "احمد" in result
        assert "2024" in result

    def test_arabic_tatweel_removed(self):
        from nlp_engine.preprocessing.normalizer import normalize
        text = "مرحـبا"
        result = normalize(text)
        assert result == "مرحبا"


class TestTokenizer:
    def test_basic_tokenize(self):
        from nlp_engine.preprocessing.tokenizer import tokenize
        tokens = tokenize("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_no_stopwords(self):
        from nlp_engine.preprocessing.tokenizer import tokenize_no_stopwords
        tokens = tokenize_no_stopwords("what are the fees for cs")
        # "what", "are", "the", "for" are stop words
        assert "fees" in tokens
        assert "cs" in tokens

    def test_empty_tokenize(self):
        from nlp_engine.preprocessing.tokenizer import tokenize
        assert tokenize("") == []

    def test_arabic_tokenize(self):
        from nlp_engine.preprocessing.tokenizer import tokenize
        tokens = tokenize("مرحبا بكم في اختبار")
        assert "مرحبا" in tokens
        assert "اختبار" in tokens

    def test_arabic_tokenize_no_stopwords(self):
        from nlp_engine.preprocessing.tokenizer import tokenize_no_stopwords
        tokens = tokenize_no_stopwords("من فضلك اريد رسوم التسجيل في قسم الحاسوب")
        assert "رسوم" in tokens
        assert "من" not in tokens
        assert "فضلك" not in tokens
        assert "اريد" not in tokens

    def test_light_arabic_stem(self):
        from nlp_engine.preprocessing.tokenizer import light_arabic_stem
        stemmed = light_arabic_stem("والطلاب في الجامعة")
        tokens = stemmed.split()
        assert "طلاب" in tokens


# ---------------------------------------------------------------------------
# Intent label tests
# ---------------------------------------------------------------------------

class TestIntentLabels:
    def test_labels_file_exists(self):
        import json
        from pathlib import Path
        labels_path = Path("nlp_engine/intent/labels.json")
        assert labels_path.exists(), "labels.json not found"
        labels = json.loads(labels_path.read_text())
        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_required_intents_present(self):
        import json
        from pathlib import Path
        labels = json.loads(Path("nlp_engine/intent/labels.json").read_text())
        required = {"faq_fees", "faq_registration", "greeting", "unknown"}
        assert required.issubset(set(labels))


# ---------------------------------------------------------------------------
# NER tests
# ---------------------------------------------------------------------------

class TestNERExtractor:
    def test_course_code_extraction(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("I want to register for CS401")
        types = [e["type"] for e in entities]
        assert EntityType.COURSE_CODE in types

    def test_student_id_extraction(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("My student ID is 1201234")
        types = [e["type"] for e in entities]
        assert EntityType.STUDENT_ID in types

    def test_department_extraction(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("What are the computer science department requirements?")
        types = [e["type"] for e in entities]
        assert EntityType.DEPARTMENT in types

    def test_semester_extraction(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("Registration opens in spring 2025")
        types = [e["type"] for e in entities]
        assert EntityType.SEMESTER in types

    def test_course_code_extraction_arabic(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("مادة CS401")
        types = [e["type"] for e in entities]
        assert EntityType.COURSE_CODE in types

    def test_student_id_extraction_arabic_digits(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("رقم الطالب ١٢٠٢٣٤٥")
        types = [e["type"] for e in entities]
        assert EntityType.STUDENT_ID in types

    def test_department_extraction_arabic(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("قسم هندسة الحاسوب")
        types = [e["type"] for e in entities]
        assert EntityType.DEPARTMENT in types

    def test_semester_extraction_arabic(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("موعد التسجيل للفصل الثاني")
        types = [e["type"] for e in entities]
        assert EntityType.SEMESTER in types

    def test_course_name_extraction_arabic(self):
        from nlp_engine.ner.extractor import extract_entities
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("مادة معالجة اللغات الطبيعية")
        types = [e["type"] for e in entities]
        assert EntityType.COURSE_NAME in types

    def test_entities_to_dict(self):
        from nlp_engine.ner.extractor import extract_entities, entities_to_dict
        from nlp_engine.ner.entities import EntityType
        entities = extract_entities("CS401 in spring 2025")
        d = entities_to_dict(entities)
        assert isinstance(d, dict)


# ---------------------------------------------------------------------------
# Evaluation metrics tests
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_intent_accuracy_perfect(self):
        from nlp_engine.evaluation.metrics import intent_accuracy
        preds = ["faq_fees", "greeting", "schedule_exams"]
        truth = ["faq_fees", "greeting", "schedule_exams"]
        assert intent_accuracy(preds, truth) == 1.0

    def test_intent_accuracy_partial(self):
        from nlp_engine.evaluation.metrics import intent_accuracy
        preds = ["faq_fees", "greeting"]
        truth = ["faq_fees", "unknown"]
        assert intent_accuracy(preds, truth) == 0.5

    def test_intent_accuracy_empty(self):
        from nlp_engine.evaluation.metrics import intent_accuracy
        assert intent_accuracy([], []) == 0.0

    def test_ner_f1_perfect(self):
        from nlp_engine.evaluation.metrics import ner_precision_recall_f1
        pred = [[{"type": "DEPARTMENT", "value": "Computer Science"}]]
        true = [[{"type": "DEPARTMENT", "value": "Computer Science"}]]
        scores = ner_precision_recall_f1(pred, true)
        assert scores["f1"] == 1.0

    def test_ner_f1_empty(self):
        from nlp_engine.evaluation.metrics import ner_precision_recall_f1
        scores = ner_precision_recall_f1([[]], [[]])
        assert scores["f1"] == 0.0

    def test_precision_at_k(self):
        from nlp_engine.evaluation.metrics import precision_at_k
        retrieved = [["a", "b", "c"]]
        relevant  = [["a", "c", "x"]]
        p_at_3 = precision_at_k(retrieved, relevant, k=3)
        assert p_at_3 == pytest.approx(2 / 3, rel=1e-3)

    def test_precision_at_k_empty(self):
        from nlp_engine.evaluation.metrics import precision_at_k
        assert precision_at_k([], [], k=3) == 0.0


# ---------------------------------------------------------------------------
# Dialogue tests
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_add_and_retrieve(self):
        from nlp_engine.dialogue.context_manager import ContextManager
        ctx = ContextManager(max_turns=4)
        ctx.add_turn("user", "Hello")
        ctx.add_turn("assistant", "Hi there!")
        history = ctx.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"

    def test_max_turns_respected(self):
        from nlp_engine.dialogue.context_manager import ContextManager
        ctx = ContextManager(max_turns=2)
        for i in range(5):
            ctx.add_turn("user", f"message {i}")
        assert len(ctx.get_history()) == 2

    def test_load_history(self):
        from nlp_engine.dialogue.context_manager import ContextManager
        ctx = ContextManager()
        ctx.load_history([{"role": "user", "content": "test"}])
        assert ctx.get_last_user_message() == "test"


class TestStateMachine:
    def test_greeting_handled(self):
        from nlp_engine.dialogue.state_machine import StateMachine
        sm = StateMachine()
        decision = sm.route("greeting", 0.95, {})
        assert decision.handled is True
        assert decision.reply is not None

    def test_low_confidence_handled(self):
        from nlp_engine.dialogue.state_machine import StateMachine
        sm = StateMachine(confidence_threshold=0.55)
        decision = sm.route("faq_fees", 0.3, {})
        assert decision.handled is True

    def test_rag_intent_not_handled(self):
        from nlp_engine.dialogue.state_machine import StateMachine
        sm = StateMachine()
        decision = sm.route("faq_fees", 0.92, {})
        assert decision.handled is False

    def test_unknown_intent_handled(self):
        from nlp_engine.dialogue.state_machine import StateMachine
        sm = StateMachine()
        decision = sm.route("unknown", 0.9, {})
        assert decision.handled is True


class TestLanguageDetection:
    def test_detect_language_arabic(self):
        from nlp_engine.preprocessing.tokenizer import detect_language
        assert detect_language("أنا أحب البرمجة") == "arabic"

    def test_detect_language_english(self):
        from nlp_engine.preprocessing.tokenizer import detect_language
        assert detect_language("How much are the fees?") == "english"


# ---------------------------------------------------------------------------
# Extended metrics tests (new functions: Recall@k, MRR, NDCG, per-class F1)
# ---------------------------------------------------------------------------

class TestMetricsExtended:
    def test_per_class_f1_perfect(self):
        from nlp_engine.evaluation.metrics import per_class_intent_f1
        preds  = ["faq_fees", "greeting", "faq_fees"]
        truth  = ["faq_fees", "greeting", "faq_fees"]
        result = per_class_intent_f1(preds, truth)
        assert result["faq_fees"]["f1"] == 1.0
        assert result["greeting"]["f1"] == 1.0

    def test_per_class_f1_empty(self):
        from nlp_engine.evaluation.metrics import per_class_intent_f1
        assert per_class_intent_f1([], []) == {}

    def test_macro_f1(self):
        from nlp_engine.evaluation.metrics import per_class_intent_f1, macro_f1
        assert macro_f1(per_class_intent_f1(["a", "b"], ["a", "b"])) == 1.0

    def test_recall_at_k_all_relevant(self):
        from nlp_engine.evaluation.metrics import recall_at_k
        assert recall_at_k([["a", "b", "c"]], [["a", "c"]], k=3) == pytest.approx(1.0, rel=1e-3)

    def test_recall_at_k_none_relevant(self):
        from nlp_engine.evaluation.metrics import recall_at_k
        assert recall_at_k([["x", "y", "z"]], [["a", "b"]], k=3) == 0.0

    def test_recall_at_k_empty(self):
        from nlp_engine.evaluation.metrics import recall_at_k
        assert recall_at_k([], [], k=3) == 0.0

    def test_mrr_first_rank(self):
        from nlp_engine.evaluation.metrics import mean_reciprocal_rank
        assert mean_reciprocal_rank([["a", "b", "c"]], [["a"]]) == 1.0

    def test_mrr_second_rank(self):
        from nlp_engine.evaluation.metrics import mean_reciprocal_rank
        assert mean_reciprocal_rank([["x", "a", "b"]], [["a"]]) == pytest.approx(0.5, rel=1e-3)

    def test_mrr_miss(self):
        from nlp_engine.evaluation.metrics import mean_reciprocal_rank
        assert mean_reciprocal_rank([["x", "y"]], [["a"]]) == 0.0

    def test_mrr_empty(self):
        from nlp_engine.evaluation.metrics import mean_reciprocal_rank
        assert mean_reciprocal_rank([], []) == 0.0

    def test_ndcg_perfect(self):
        from nlp_engine.evaluation.metrics import ndcg_at_k
        assert ndcg_at_k([["a", "b", "c"]], [["a", "b", "c"]], k=3) == pytest.approx(1.0, rel=1e-3)

    def test_ndcg_empty(self):
        from nlp_engine.evaluation.metrics import ndcg_at_k
        assert ndcg_at_k([], [], k=3) == 0.0


# ---------------------------------------------------------------------------
# State machine v2 — canned-before-confidence-check
# ---------------------------------------------------------------------------

class TestStateMachineV2:
    def test_greeting_passes_at_low_confidence(self):
        from nlp_engine.dialogue.state_machine import StateMachine
        sm = StateMachine(confidence_threshold=0.55)
        decision = sm.route("greeting", 0.40, {}, query="\u0645\u0631\u062d\u0628\u0627")
        assert decision.handled is True
        assert decision.reason.startswith("canned:")

    def test_thanks_passes_at_low_confidence(self):
        from nlp_engine.dialogue.state_machine import StateMachine
        sm = StateMachine(confidence_threshold=0.55)
        decision = sm.route("thanks", 0.45, {}, query="\u0634\u0643\u0631\u0627")
        assert decision.handled is True
        assert decision.reason.startswith("canned:")

    def test_low_confidence_non_canned_asks_clarification(self):
        from nlp_engine.dialogue.state_machine import StateMachine
        sm = StateMachine(confidence_threshold=0.55)
        decision = sm.route("faq_fees", 0.30, {})
        assert decision.handled is True
        assert decision.reason == "low_confidence"


# ---------------------------------------------------------------------------
# Shared language utility (shared.utils.lang)
# ---------------------------------------------------------------------------

class TestSharedIsArabic:
    def test_arabic_text(self):
        from shared.utils.lang import is_arabic
        assert is_arabic("\u0645\u0631\u062d\u0628\u0627") is True

    def test_english_text(self):
        from shared.utils.lang import is_arabic
        assert is_arabic("Hello") is False

    def test_mixed_text(self):
        from shared.utils.lang import is_arabic
        assert is_arabic("CS401 \u0641\u064a \u0627\u0644\u062c\u0627\u0645\u0639\u0629") is True

    def test_empty_string(self):
        from shared.utils.lang import is_arabic
        assert is_arabic("") is False
