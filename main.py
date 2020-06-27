import pkgutil
import sys
import video 
import answer

def main():
	"""
		This is the script wil be used to execute your solution. Make sure your
		output format is the same as instructed in the GitHub repo.
		Notice that the input format should not be changed. The following commands will be used
		to run your solution:
		$ unzip solution.pyz -d solution/
		$ cd solution/
		$ pip3 install -r requirements.txt
		$ cd ../
		$ python3 solution/ video.mp4 questions.txt
		Only the last line will be monitored by the power meter for efficiency and performance.
	"""
	text=video.detect(sys.argv[1])
	answer.output_answer_from_list(text,sys.argv[2])

