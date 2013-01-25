// manual beat labeling in Go for faster speed
// input filename should be sampled at 48000 KHz
package main

import (
	"flag"
	"fmt"
	"log"
	"os/exec"
	"time"
)

func main() {
	var inputfilename = flag.String("input-filename", "", "the input filename")
	var offset = flag.Float64("offset", 0.25, "offset subtracted in seconds to account for keyboard lag")
	flag.Parse()
	// start the timer
	var cmd = exec.Command("aplay", *inputfilename)
	e := cmd.Start()
	if e != nil {
		log.Fatal(e)
	}
	var starttime = time.Now()
	var q string
	var elapsed time.Duration
	beats := make([]int, 0)
	for {
		fmt.Scanf("%s", &q)
		elapsed = time.Since(starttime)
		if q == "q" {
			break
		}
		// move the beats back to account for keyboard lag
		// an offset of 0.25 works for some users, but feel free to customize
		beats = append(beats, int((elapsed.Seconds() - *offset) * 48000.))
	}
	// kill the audio if it's still playing
	cmd.Process.Kill()
	fmt.Println(beats)
}
