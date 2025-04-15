

cap = cv2.VideoCapture(2)
showtext = "press n to predict"
while(1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.putText(frame, showtext, (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 200, 200), 2)

    cv2.imshow("capture", frame)

    keyvalue = cv2.waitKey(1)

    if keyvalue == 27:
        break
    elif keyvalue == ord('n'):
        break

cap.release()
cv2.destroyAllWindows()