from .constants import (
    NAIVE_CHECK_PREFIXES,
    NAIVE_CHECK_ALLOWED_CHARS,
    NAIVE_CHECK_KEYWORDS,
)

from abc import abstractmethod

import string

##########################
# Clause Checker
##########################


class ClauseChecker:
    @staticmethod
    @abstractmethod
    def check(clause_s):
        pass


class NaiveChecker(ClauseChecker):
    @staticmethod
    def check(clause_s):
        clause_s = clause_s.lower().strip()

        # check EOS
        if "</s>" in clause_s or "<s>" in clause_s:
            return False
        if "<|im_end|>" in clause_s or "<|im_start|>" in clause_s:
            return False

        # check isascii
        if not clause_s.isascii():
            return False

        # check punctuations
        if (
            all(c in (string.punctuation + string.whitespace) for c in clause_s)
            and len(clause_s) > 3
        ):
            return False

        # check prefix and keyword
        s = "".join(c for c in clause_s if c in NAIVE_CHECK_ALLOWED_CHARS)
        s = s.strip()  # after filtering, there may be new whitespaces

        # check prefix
        for prefix in NAIVE_CHECK_PREFIXES:
            if s.startswith(prefix):
                return False

        # check keyword
        s += " "  # the space is important to ensure the keyword is a word
        for keyword in NAIVE_CHECK_KEYWORDS:
            if keyword in s:
                return False

        return True
