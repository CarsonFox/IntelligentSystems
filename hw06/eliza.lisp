;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-

#|
=========================================================
Module: eliza.lisp: 
Description: A version of ELIZA that takes inputs without 
paretheses around them unlike eliza1.lisp.
Bugs to vladimir kulyukin in canvas
=========================================================
|#

;;; ==============================

(defun rule-pattern (rule) (first rule))
(defun rule-responses (rule) (rest rule))

(defun read-line-no-punct ()
  "Read an input line, ignoring punctuation."
  (read-from-string
    (concatenate 'string "(" (substitute-if #\space #'punctuation-p
                                            (read-line))
                 ")")))

(defun punctuation-p (char) (find char ".,;:`!?#-()\\\""))

;;; ==============================

(defun use-eliza-rules (input)
  "Find some rule with which to transform the input."
  (some #'(lambda (rule)
            (let ((result (pat-match (rule-pattern rule) input)))
              (if (not (eq result fail))
                  (sublis (switch-viewpoint result)
                          (random-elt (rule-responses rule))))))
        *eliza-rules*))

(defun switch-viewpoint (words)
  "Change I to you and vice versa, and so on."
  (sublis '((i . you) (you . i) (me . you) (am . are) (my . your) (your . my))
          words))

(defparameter *good-byes* '((good bye) (see you) (see you later) (so long)))

(defun eliza ()
  "Respond to user input using pattern matching rules."
  (loop
    (print 'eliza>)
    (let* ((input (read-line-no-punct))
           (response (flatten (use-eliza-rules input))))
      (print-with-spaces response)
      (if (member response *good-byes* :test #'equal)
	  (RETURN))))
  (values))

(defun print-with-spaces (list)
  (mapc #'(lambda (x) (prin1 x) (princ " ")) list))

(defun print-with-spaces (list)
  (format t "~{~a ~}" list))

;;; ==============================

(defparameter *eliza-rules*
  '(
    ;;; rule 1
    (((?* ?x) hello (?* ?y))      
    (How do you do.  Please state your problem.))

    ;;; rule 2
    (((?* ?x) computer (?* ?y))
     (Do computers worry you?)
     (What do you think about machines?)
     (Why do you mention computers?)
     (What do you think machines have to do with your problem?))

    ;;; rule 3
    (((?* ?x) name (?* ?y))
     (I am not interested in names))

    ;;; rule 4
    (((?* ?x) sorry (?* ?y))
     (Please don't apologize)
     (Apologies are not necessary)
     (What feelings do you have when you apologize))

    ;;; rule 5
    (((?* ?x) remember (?* ?y)) 
     (Do you often think of ?y)
     (Does thinking of ?y bring anything else to mind?)
     (What else do you remember)
     (Why do you recall ?y right now?)
     (What in the present situation reminds you of ?y)
     (What is the connection between me and ?y))

    ;;; rule 6
    (((?* x) good bye (?* y))
     (good bye))

    ;;; rule 7
    ;;;(((?* x) so long (?* y))
    ;;; (good bye)
    ;;; (bye)
    ;;; (see you)
    ;;; (see you later))

    ;;; ========== your rules begin
    ;;; add your rules here
    ;;; ========== your rules end

    ;;; rule 8
    (((?* ?z) am afraid of (?* ?x))
     (When did you first notice you were afraid of ?x)
     (What caused your fear of ?x)
     (Why are you so scared of ?x))

    ;;; rule 9
    (((?* ?x) want to (?* ?y))
     (Why do you think you want to ?y)
     (What will it accomplish to ?y)
     (After you ?y what will you do?))

    ;;; rule 10
    (((?* ?z) hate (?* ?x))
      (What do you hate about ?x)
      (Maybe ?x isn't as bad as you think)
      (Well I happen to _LOVE_ ?x)
      (?x hates you too))

    ;;; rule 11
    (((?* ?x) think that (?* ?y))
     (why do you feel that way?)
     (do you think that is a popular opinion?)
     (I don't think so.))

    ;;; rule 12
    (((?* ?x) cannot (?* ?y))
     (maybe ?x were never meant to ?y)
     (?x do not want to ?y anyways)
     (what needs to happen before ?x can ?y))

    ;;; rule 13
    (((?* ?x) will (?* ?y))
     (when will ?x ?y)
     (why would ?x do that?)
     (how will you feel when that happens?))

    ;;; rule 14
    (((?* ?x) feel (?* ?y))
     (when do ?x feel that way?)
     (what makes ?x feel ?y)
     (what can ?x do about it?))

    ;;; rule 15
    (((?* ?x) have to (?* ?y))
     (why do ?x have to ?y)
     (when will ?x ?y)
     (how will ?x ?y)
     (what do ?x hope to accomplish when ?x ?y))

    ;;; rule 16
    (((?* ?x) is (?* ?y) favorite (?* ?z))
     (what is ?y second-favorite ?z)
     (what awful taste in ?z)
     (then what is the worst ?z)
     (?x is not that nice)
     (why is ?x your favorite?))

    ;;; rule 17
    ((because (?* ?x))
     (that is a bad reason)
     (is that what you really think?)
     (that is not true))
   ))

;;; ==============================

