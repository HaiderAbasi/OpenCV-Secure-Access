activate_sa()
|
|- 1> Variables initalized
|- 2> Read Video 
|- 3> while (cap.isOpened())
|
|---- 4> if Frame % dth_frame
|-------- a> Hog-Face-detection()
|
|---- 5> if tracker-in detection mode:
|--------- a> Found faces ?
|------------- i) preprocess()
|------------- ii) track()
|                  ---         ---
|------------- iii) |-recognize-| ()
|
|---- 6> else:
|
|--------- if detected more faces ? - compare-with-tracked()
|------------  if matched()-> set to recognize
|------------  else()-> append to new-bbs 
|
|--------- if new-bbs found()
|-------------- initalize-track-new-bbs()
|               ---         ---
|--------------  |-recognize-| ()
|
|---------- else()
|-------------- if not-recognizing-yet()
|-------------------  preprocess()
|                    ---         ---
|-------------------  |-recognize-| ()
|
|-------------- identify()
|-------------- track()