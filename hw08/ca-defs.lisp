#|
==============================================
-*- Mode: Lisp; Syntax: Common-Lisp -*-

File: ca-defs.lisp
Author: Vladimir Kulyukin
Description: Simple word definitions for CA
===============================================
|#

(in-package :user)

(define-ca-word 
    jack
    (concept nil (human :name (jack) :sex (male))))

(define-ca-word
    john
    (concept nil (human :name (john) :sex (male))))

(define-ca-word
    ann
    (concept nil (human :name (ann) :sex (female))))

(define-ca-word 
    ate
  (concept ?act (ingest :time (past)))
  (request (test (before ?act ?actor (animate)))
           (actions (modify ?act :actor ?actor)))
  (request (test (after ?act ?food (food)))
	   (actions (modify ?act :object ?food))))

(define-ca-word
    bought
    (concept ?act (atrans :time (past)))
    (request (test (before ?act ?actor (animate)))
	     (actions (modify ?act :actor ?actor)))
    (request (test (after ?act ?obj (phys-obj)))
	     (actions (modify ?act :object ?obj))))

(define-ca-word 
    apple
    (concept nil (apple)))

(define-ca-word
    pear
    (concept nil (pear)))

(define-ca-word 
    an
    (mark ?x)
    (request (test (after ?x ?con (concept)))
	     (actions (modify ?con :ref (indef))))
    (request (test (after ?x ?con (concept)))
	     (actions (modify ?con :number (singular)))))

(define-ca-word
    a
    (mark ?x)
    (request (test (after ?x ?loc (location)))
             (actions (modify ?loc :ref (indef))))
    (request (test (after ?x ?loc (location)))
             (actions (modify ?loc :number (singular))))
    (request (test (after ?x ?obj (phys-obj)))
	     (actions (modify ?obj :ref (indef))))
    (request (test (after ?x ?obj (phys-obj)))
	     (actions (modify ?obj :number (singular)))))

(define-ca-word
    went
    ;;; your code here
    )

(define-ca-word
    restaurant
    ;;; your code here
    )

(define-ca-word
    home
    ;;; your code here
    )

(define-ca-word
    to
    ;;; your code here
    )


(define-ca-word
    lobster
    ;;; your code here
    )

;;; end-of-file

