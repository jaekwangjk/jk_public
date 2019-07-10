 /*  FFmpeg arguments */
 
 
    // -y	Tells FFmpeg to overwrite the output file if it already exists.
    
    // -f rawvideo	Sets the input format as raw video data. I’m not too sure about the relationship between this option and the next one!
    
    // -vcodec rawvideo	Tells FFmpeg to interpret its input as raw video data (i.e. unencoded frames of plain pixels).
    // -pix_fmt rgb24	Sets the input pixel format to 3-byte RGB pixels – one byte for red, one for blue and one for green.
    
    // -s n1xn2	Sets the frame size to n1×n2 pixels. FFmpeg will form the incoming pixel data into frames of this size.
    
    // -r 25	Sets the frame rate of the incoming data to 25 frames per second.
    
    // -i -	Tells FFmpeg to read its input from stdin, which means it will be reading the data out C program writes to its output pipe.
    
    // -f mp4	Sets the output file format to MP4.
    
    // -q:v 5	This controls the quality of the encoded MP4 file. The numerical range for this option is from 1 (highest quality, biggest file size) to 32 (lowest quality, smallest file size). I arrived at a value of 5 by trial and error. Subjectively, it seemed to me to give roughly the best trade off between file size and quality.
    
    // -an	Specifies no audio stream in the output file.
    
    // -vcodec mpeg4	Tells FFmpeg to use its “mpeg4” encoder.
    
    // out.mp4	Specifies out.mp4 as the output file.
 /*  end ffmepg arguments */
