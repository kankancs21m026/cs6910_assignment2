from gtts import gTTS
import os
def play(mytext) :
    # Language in which you want to convert
    language = 'en'
  
    # Passing the text and language to the engine, 
    myobj = gTTS(text=mytext, lang=language, slow=False)
  
    # Saving the converted audio in a mp3 file named
    # welcome 
    myobj.save("op.mp3")
  
    # Playing the converted file
    os.system("start op.mp3")