import requests
import lkh




problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text
problem_str = """SPECIAL
NAME: gr17
TYPE: SOP
COMMENT: 17-city problem (Groetschel)
DIMENSION: 17
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
17
 0 633 0 257 390 0 91 661 228 0 412 227
 169 383 0 150 488 112 120 267 0 80 572 196
 77 351 63 0 134 530 154 105 309 34 29 0
 259 555 372 175 338 264 232 249 0 505 289 262
 476 196 360 444 402 495 0 353 282 110 324 61
 208 292 250 352 154 0 324 638 437 240 421 329
 297 314 95 578 435 0 70 567 191 27 346 83
 47 68 189 439 287 254 0 211 466 74 182 243
 105 150 108 326 336 184 391 145 0 268 420 53
 239 199 123 207 165 383 240 140 448 202 57 0
 246 745 472 237 528 364 332 349 202 685 542 157
 289 426 483 0 121 518 142 84 297 35 29 36
 -1 390 238 301 55 96 153 336 0 
EOF
"""
#problem_str = requests.get('http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/A/A-n32-k5.vrp').text

problem = lkh.LKHProblem.parse(problem_str)
# Path to the .sop file
sop_file_path = "C:/Users/elisa/Downloads/SOP/SOP/INSTANCES/TSPLIB/ESC07.sop"

# Open the .sop file and read it
with open(sop_file_path, "r") as sop_file:
    problem = lkh.LKHProblem.read(sop_file)
print(problem)
solver_path = "C:/Users/elisa/Downloads/LKH-3.exe" #perhaps it would be better to use a Docker container for the LKH installation
solution = lkh.solve(solver_path, problem=problem, time_limit=120, runs = 100,MOVE_TYPE = "5 SPECIAL",GAIN23 = "NO",KICKS = 1,MAX_SWAPS = 0,POPULATION_SIZE = 10,KICK_TYPE=4)
print(solution)