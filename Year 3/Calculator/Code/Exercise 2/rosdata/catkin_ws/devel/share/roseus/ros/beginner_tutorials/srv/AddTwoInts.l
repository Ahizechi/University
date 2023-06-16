;; Auto-generated. Do not edit!


(when (boundp 'beginner_tutorials::AddTwoInts)
  (if (not (find-package "BEGINNER_TUTORIALS"))
    (make-package "BEGINNER_TUTORIALS"))
  (shadow 'AddTwoInts (find-package "BEGINNER_TUTORIALS")))
(unless (find-package "BEGINNER_TUTORIALS::ADDTWOINTS")
  (make-package "BEGINNER_TUTORIALS::ADDTWOINTS"))
(unless (find-package "BEGINNER_TUTORIALS::ADDTWOINTSREQUEST")
  (make-package "BEGINNER_TUTORIALS::ADDTWOINTSREQUEST"))
(unless (find-package "BEGINNER_TUTORIALS::ADDTWOINTSRESPONSE")
  (make-package "BEGINNER_TUTORIALS::ADDTWOINTSRESPONSE"))

(in-package "ROS")





(defclass beginner_tutorials::AddTwoIntsRequest
  :super ros::object
  :slots (_first _tenth ))

(defmethod beginner_tutorials::AddTwoIntsRequest
  (:init
   (&key
    ((:first __first) "")
    ((:tenth __tenth) 0.0)
    )
   (send-super :init)
   (setq _first (string __first))
   (setq _tenth (float __tenth))
   self)
  (:first
   (&optional __first)
   (if __first (setq _first __first)) _first)
  (:tenth
   (&optional __tenth)
   (if __tenth (setq _tenth __tenth)) _tenth)
  (:serialization-length
   ()
   (+
    ;; string _first
    4 (length _first)
    ;; float64 _tenth
    8
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _first
       (write-long (length _first) s) (princ _first s)
     ;; float64 _tenth
       (sys::poke _tenth (send s :buffer) (send s :count) :double) (incf (stream-count s) 8)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _first
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _first (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;; float64 _tenth
     (setq _tenth (sys::peek buf ptr- :double)) (incf ptr- 8)
   ;;
   self)
  )

(defclass beginner_tutorials::AddTwoIntsResponse
  :super ros::object
  :slots (_sum ))

(defmethod beginner_tutorials::AddTwoIntsResponse
  (:init
   (&key
    ((:sum __sum) "")
    )
   (send-super :init)
   (setq _sum (string __sum))
   self)
  (:sum
   (&optional __sum)
   (if __sum (setq _sum __sum)) _sum)
  (:serialization-length
   ()
   (+
    ;; string _sum
    4 (length _sum)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; string _sum
       (write-long (length _sum) s) (princ _sum s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; string _sum
     (let (n) (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4) (setq _sum (subseq buf ptr- (+ ptr- n))) (incf ptr- n))
   ;;
   self)
  )

(defclass beginner_tutorials::AddTwoInts
  :super ros::object
  :slots ())

(setf (get beginner_tutorials::AddTwoInts :md5sum-) "70e914cd01f6d2dbd6db72500ab2b805")
(setf (get beginner_tutorials::AddTwoInts :datatype-) "beginner_tutorials/AddTwoInts")
(setf (get beginner_tutorials::AddTwoInts :request) beginner_tutorials::AddTwoIntsRequest)
(setf (get beginner_tutorials::AddTwoInts :response) beginner_tutorials::AddTwoIntsResponse)

(defmethod beginner_tutorials::AddTwoIntsRequest
  (:response () (instance beginner_tutorials::AddTwoIntsResponse :init)))

(setf (get beginner_tutorials::AddTwoIntsRequest :md5sum-) "70e914cd01f6d2dbd6db72500ab2b805")
(setf (get beginner_tutorials::AddTwoIntsRequest :datatype-) "beginner_tutorials/AddTwoIntsRequest")
(setf (get beginner_tutorials::AddTwoIntsRequest :definition-)
      "string first
float64 tenth
---
string sum

")

(setf (get beginner_tutorials::AddTwoIntsResponse :md5sum-) "70e914cd01f6d2dbd6db72500ab2b805")
(setf (get beginner_tutorials::AddTwoIntsResponse :datatype-) "beginner_tutorials/AddTwoIntsResponse")
(setf (get beginner_tutorials::AddTwoIntsResponse :definition-)
      "string first
float64 tenth
---
string sum

")



(provide :beginner_tutorials/AddTwoInts "70e914cd01f6d2dbd6db72500ab2b805")


