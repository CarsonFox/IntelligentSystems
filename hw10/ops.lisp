;;; -*- Mode: Lisp; Syntax: Common-Lisp; -*-
;;; Module: ops.lisp
;;; different worlds and operators for the GPS planner.
;;; bugs to vladimir kulyukin in canvas
;;; =========================================

;;; Problem 2 plan:
;;; ((START) (EXECUTE MOVE-C-A->T) (EXECUTE MOVE-B-T->C) (EXECUTE MOVE-A-T->B))

;;; Problem 3 plan:
;;; ((START) (EXECUTE DROP-BALL) (EXECUTE PUSH-CHAIR) (EXECUTE CLIMB-CHAIR) (EXECUTE GRASP-BANANA) (EXECUTE EAT-BANANA))

(in-package :user)

(defstruct op "An operation"
  (action nil) 
  (preconds nil) 
  (add-list nil) 
  (del-list nil))

(defun executing-p (x)
  "Is x of the form: (execute ...) ?"
  (starts-with x 'execute))

(defun convert-op (op)
  "Make op conform to the (EXECUTING op) convention."
  (unless (some #'executing-p (op-add-list op))
    (push (list 'execute (op-action op)) (op-add-list op)))
  op)

(defun op (action &key preconds add-list del-list)
  "Make a new operator that obeys the (EXECUTING op) convention."
  (convert-op
    (make-op :action action :preconds preconds
             :add-list add-list :del-list del-list)))

;;; ================= Son At School ====================

(defparameter *school-world* '(son-at-home car-needs-battery
					   have-money have-phone-book))

(defparameter *school-ops*
  (list
    ;;; operator 1
   (make-op :action 'drive-son-to-school
	    :preconds '(son-at-home car-works)
	    :add-list '(son-at-school)
	    :del-list '(son-at-home))
   ;;; operator 2
   (make-op :action 'shop-installs-battery
	    :preconds '(car-needs-battery shop-knows-problem shop-has-money)
	    :add-list '(car-works))
   ;;; operator 3
   (make-op :action 'tell-shop-problem
	    :preconds '(in-communication-with-shop)
	    :add-list '(shop-knows-problem))
   ;;; operator 4
   (make-op :action 'telephone-shop
	    :preconds '(know-phone-number)
	    :add-list '(in-communication-with-shop))
   ;;; operator 5
   (make-op :action 'look-up-number
	    :preconds '(have-phone-book)
	    :add-list '(know-phone-number))
   ;;; operator 6
   (make-op :action 'give-shop-money
	    :preconds '(have-money)
	    :add-list '(shop-has-money)
	    :del-list '(have-money))))

;;; ================= Sussman's Anomaly ====================

(defparameter *block-world* '(a-on-t b-on-t c-on-a clear-c clear-b))

(defparameter *block-ops*
  (list
    (make-op :action 'move-c-a->t
             :preconds '(c-on-a clear-c)
             :add-list '(clear-a a-on-t)
             :del-list '(c-on-a))
    (make-op :action 'move-b-t->c
             :preconds '(b-on-t clear-b clear-c)
             :add-list '(b-on-c)
             :del-list '(clear-c b-on-t))
    (make-op :action 'move-a-t->b
             :preconds '(a-on-t clear-a clear-b b-on-c)
             :add-list '(a-on-b)
             :del-list '(a-on-t clear-b))
   )
  )
	    
;;; ================= Monkey and Bananas ====================

(defparameter *banana-world* '(at-door on-floor has-ball hungry chair-at-door))

(defparameter *banana-ops*
  (list
    (make-op :action 'eat-banana
             :preconds '(grasped-banana)
             :add-list '(not-hungry)
             :del-list '(hungry grasped-banana))
    (make-op :action 'grasp-banana
             :preconds '(climbed-chair emptyhanded)
             :add-list '(grasped-banana))
    (make-op :action 'climb-chair
             :preconds '(pushed-chair)
             :del-list '(on-floor)
             :add-list '(climbed-chair))
    (make-op :action 'push-chair
             :preconds '(chair-at-door at-door emptyhanded)
             :add-list '(pushed-chair)
             :del-list '(chair-at-door at-door))
    (make-op :action 'drop-ball
             :preconds '(has-ball on-floor)
             :add-list '(emptyhanded)
             :del-list '(has-ball))
    )
  )
  
(mapc #'convert-op *school-ops*)
(mapc #'convert-op *block-ops*)
(mapc #'convert-op *banana-ops*)

(provide :ops)
