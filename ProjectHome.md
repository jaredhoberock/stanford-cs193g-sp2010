# Course Objectives #

Virtually all semiconductor market domains, including PCs, game consoles, mobile handsets, servers, supercomputers, and networks, are converging to concurrent platforms. There are two important reasons for this trend. First, these concurrent processors can potentially offer more effective use of chip space and power than traditional monolithic microprocessors for many demanding applications. Second, an increasing number of applications that traditionally used Application Specific Integrated Circuits (ASICs) are now implemented with concurrent processors in order to improve functionality and reduce engineering cost. The real challenge is to develop applications software that effectively uses these concurrent processors to achieve efficiency and performance goals.


The aim of this course is to provide students with knowledge and hands-on experience in developing applications software for processors with massively parallel computing resources. In general, we refer to a processor as massively parallel if it has the ability to complete more than 64 arithmetic operations per clock cycle. Many commercial offerings from NVIDIA, AMD, and Intel already offer such levels of concurrency. Effectively programming these processors will require in-depth knowledge about parallel programming principles, as well as the parallelism models, communication models, and resource limitations of these processors. The target audiences of the course are students who want to develop exciting applications for these processors, as well as those who want to develop programming tools and future implementations for these processors.


We will be using NVIDIA processors and the CUDA programming tools in the lab section of the course. Many have reported success in performing non-graphics parallel computation as well as traditional graphics rendering computation on these processors. You will go through structured programming assignments before being turned loose on the final project. Each programming assignment will involve successively more sophisticated programming skills. The final project will be of your own design, with the requirement that the project must involve a demanding application such as mathematics- or physics-intensive simulation or other data-intensive computation, followed by some form of visualization and display of results.

# Contact Information #
Instructors:<br>
Jared Hoberock. email: jaredhoberock (at) gmail.com<br>
David Tarjan. email: tar.cs193g (at) gmail.com<br>
<br>
Teaching Assistant:<br>
Niels Joubert. email: njoubert (at) cs.stanford.edu<br>

<a href='http://groups.google.com/group/cs193g-discuss'>Class mailing list</a>

<h1>Lecture and Office Hours</h1>
Lecture: 4:15-5:30 PM, Tu Th, Thornton 102.<br>
Instructor Office Hours: 3:00-4:00 PM, Tu Th, Gates 195.<br>
TA Office Hours: 1:00-2:00 PM, M,W,F, Gates 268<br>
<br>
<h1>Textbook & Materials</h1>

Textbook: Kirk & Hwu, <a href='http://www.amazon.com/Programming-Massively-Parallel-Processors-Hands-/dp/0123814723%3FSubscriptionId%3DAKIAIMLFGT27G534DNLQ%26tag%3Dhi06-20%26linkCode%3Dxm2%26camp%3D2025%26creative%3D165953%26creativeASIN%3D0123814723'>"Programming Massively Parallel Processors: A Hands-on Approach"</a>.<br>
<br>
Online materials <a href='http://courses.ece.illinois.edu/ece498/al/Syllabus.html'>available here</a>.<br>
<br>
CUDA reference materials:<br>
<ul><li><a href='http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/NVIDIA_CUDA_ProgrammingGuide.pdf'>NVIDIA CUDA Programming Guide</a>
</li><li><a href='http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/CudaReferenceManual.pdf'>CUDA Reference Manual</a>
</li><li><a href='http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/docs/online/index.html'>CUDA Online Reference</a></li></ul>

<h1>Acknowledgements</h1>

This course is based on Wen-mei Hwu & David Kirk's <a href='http://courses.ece.illinois.edu/ece498/al/'>UIUC ECE 498 AL: Applied Parallel Programming</a> class.  We appreciate their generosity in providing their course materials to others.