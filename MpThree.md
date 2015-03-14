# Machine Problem Three #

In MP3, we'll explore compute divergence as a performance hazard in the context of the [Black-Scholes](http://en.wikipedia.org/wiki/Black-scholes) algorithm, a well-known computationally-bound kernel.  We'll build a simple divergence-avoiding schedule based on stream compaction, itself implemented with the parallel patterns scan and scatter.

The third MP consists of three parts. You can get all the files you'll need by doing an svn checkout from our source tree as follows:

```
svn checkout http://stanford-cs193g-sp2010.googlecode.com/svn/trunk/assignments/mp3 mp3
```