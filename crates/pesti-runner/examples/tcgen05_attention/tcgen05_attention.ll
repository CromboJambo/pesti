; ModuleID = 'builtin.module'
source_filename = "tcgen05_attention"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_2 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_1 = addrspace(3) global [1 x i64] zeroinitializer, align 8
@__shared_mem_0 = addrspace(3) global [1 x i32] zeroinitializer, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() #0
declare void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3), i32) #0

define ptx_kernel void @tcgen05_attention_kernel(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, float %v12) {
entry:
  %v13 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v5, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v7, 1
  br label %bb0
bb0:
  %v21 = phi { ptr, i64 } [ %v14, %entry ]
  %v22 = phi { ptr, i64 } [ %v16, %entry ]
  %v23 = phi { ptr, i64 } [ %v18, %entry ]
  %v24 = phi { ptr, i64 } [ %v20, %entry ]
  %v25 = phi i32 [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi i32 [ %v10, %entry ]
  %v28 = phi i32 [ %v11, %entry ]
  %v29 = phi float [ %v12, %entry ]
  %v30 = alloca [128 x float]
  %v31 = alloca [128 x float]
  %v32 = zext i32 %v25 to i64
  %v33 = zext i32 %v26 to i64
  %v34 = zext i32 %v27 to i64
  %v35 = zext i32 %v28 to i64
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v37 = zext i32 %v36 to i64
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb87
bb2:
  %v39 = zext i32 %v361 to i64
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v41 = zext i32 %v40 to i64
  %v42 = icmp uge i64 %v41, %v33
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb5, label %bb4
bb4:
  br label %bb85
bb5:
  %v44 = mul i64 %v41, %v32
  %v45 = mul i64 %v33, %v32
  %v46 = icmp ult i64 %v34, 128
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb7, label %bb6
bb6:
  br label %bb8
bb7:
  br label %bb8
bb8:
  %v48 = phi i64 [ %v34, %bb6 ], [ 128, %bb7 ]
  %v49 = icmp ult i64 %v35, 128
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb10, label %bb9
bb9:
  br label %bb11
bb10:
  br label %bb11
bb11:
  %v51 = phi i64 [ %v35, %bb9 ], [ 128, %bb10 ]
  %v52 = icmp eq i64 %v48, 0
  br i1 %v52, label %bb14, label %bb12
bb12:
  %v53 = icmp eq i64 %v51, 0
  br i1 %v53, label %bb14, label %bb13
bb13:
  %v54 = icmp eq i64 %v32, 0
  br i1 %v54, label %bb14, label %bb15
bb14:
  br label %bb85
bb15:
  %v55 = xor i1 %v359, 1
  br i1 %v55, label %bb18, label %bb16
bb16:
  %v56 = addrspacecast ptr addrspace(3) @__shared_mem_0 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v56, i32 512) #0
  br label %bb17
bb17:
  call void @llvm.nvvm.barrier0() #0
  br label %bb18
bb18:
  %v58 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v59 = addrspacecast ptr addrspace(3) %v58 to ptr
  %v60 = load i32, ptr %v59
  %v61 = xor i1 %v360, 1
  br i1 %v61, label %bb21, label %bb19
bb19:
  %v62 = trunc i64 %v39 to i32
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_1, i32 %v62) #0
  br label %bb20
bb20:
  call void @llvm.nvvm.mbarrier.init.shared(ptr addrspace(3) @__shared_mem_2, i32 %v62) #0
  br label %bb21
bb21:
  call void @llvm.nvvm.barrier0() #0
  br label %bb22
bb22:
  %v66 = insertvalue [128 x float] undef, float 0.0, 0
  %v67 = insertvalue [128 x float] %v66, float 0.0, 1
  %v68 = insertvalue [128 x float] %v67, float 0.0, 2
  %v69 = insertvalue [128 x float] %v68, float 0.0, 3
  %v70 = insertvalue [128 x float] %v69, float 0.0, 4
  %v71 = insertvalue [128 x float] %v70, float 0.0, 5
  %v72 = insertvalue [128 x float] %v71, float 0.0, 6
  %v73 = insertvalue [128 x float] %v72, float 0.0, 7
  %v74 = insertvalue [128 x float] %v73, float 0.0, 8
  %v75 = insertvalue [128 x float] %v74, float 0.0, 9
  %v76 = insertvalue [128 x float] %v75, float 0.0, 10
  %v77 = insertvalue [128 x float] %v76, float 0.0, 11
  %v78 = insertvalue [128 x float] %v77, float 0.0, 12
  %v79 = insertvalue [128 x float] %v78, float 0.0, 13
  %v80 = insertvalue [128 x float] %v79, float 0.0, 14
  %v81 = insertvalue [128 x float] %v80, float 0.0, 15
  %v82 = insertvalue [128 x float] %v81, float 0.0, 16
  %v83 = insertvalue [128 x float] %v82, float 0.0, 17
  %v84 = insertvalue [128 x float] %v83, float 0.0, 18
  %v85 = insertvalue [128 x float] %v84, float 0.0, 19
  %v86 = insertvalue [128 x float] %v85, float 0.0, 20
  %v87 = insertvalue [128 x float] %v86, float 0.0, 21
  %v88 = insertvalue [128 x float] %v87, float 0.0, 22
  %v89 = insertvalue [128 x float] %v88, float 0.0, 23
  %v90 = insertvalue [128 x float] %v89, float 0.0, 24
  %v91 = insertvalue [128 x float] %v90, float 0.0, 25
  %v92 = insertvalue [128 x float] %v91, float 0.0, 26
  %v93 = insertvalue [128 x float] %v92, float 0.0, 27
  %v94 = insertvalue [128 x float] %v93, float 0.0, 28
  %v95 = insertvalue [128 x float] %v94, float 0.0, 29
  %v96 = insertvalue [128 x float] %v95, float 0.0, 30
  %v97 = insertvalue [128 x float] %v96, float 0.0, 31
  %v98 = insertvalue [128 x float] %v97, float 0.0, 32
  %v99 = insertvalue [128 x float] %v98, float 0.0, 33
  %v100 = insertvalue [128 x float] %v99, float 0.0, 34
  %v101 = insertvalue [128 x float] %v100, float 0.0, 35
  %v102 = insertvalue [128 x float] %v101, float 0.0, 36
  %v103 = insertvalue [128 x float] %v102, float 0.0, 37
  %v104 = insertvalue [128 x float] %v103, float 0.0, 38
  %v105 = insertvalue [128 x float] %v104, float 0.0, 39
  %v106 = insertvalue [128 x float] %v105, float 0.0, 40
  %v107 = insertvalue [128 x float] %v106, float 0.0, 41
  %v108 = insertvalue [128 x float] %v107, float 0.0, 42
  %v109 = insertvalue [128 x float] %v108, float 0.0, 43
  %v110 = insertvalue [128 x float] %v109, float 0.0, 44
  %v111 = insertvalue [128 x float] %v110, float 0.0, 45
  %v112 = insertvalue [128 x float] %v111, float 0.0, 46
  %v113 = insertvalue [128 x float] %v112, float 0.0, 47
  %v114 = insertvalue [128 x float] %v113, float 0.0, 48
  %v115 = insertvalue [128 x float] %v114, float 0.0, 49
  %v116 = insertvalue [128 x float] %v115, float 0.0, 50
  %v117 = insertvalue [128 x float] %v116, float 0.0, 51
  %v118 = insertvalue [128 x float] %v117, float 0.0, 52
  %v119 = insertvalue [128 x float] %v118, float 0.0, 53
  %v120 = insertvalue [128 x float] %v119, float 0.0, 54
  %v121 = insertvalue [128 x float] %v120, float 0.0, 55
  %v122 = insertvalue [128 x float] %v121, float 0.0, 56
  %v123 = insertvalue [128 x float] %v122, float 0.0, 57
  %v124 = insertvalue [128 x float] %v123, float 0.0, 58
  %v125 = insertvalue [128 x float] %v124, float 0.0, 59
  %v126 = insertvalue [128 x float] %v125, float 0.0, 60
  %v127 = insertvalue [128 x float] %v126, float 0.0, 61
  %v128 = insertvalue [128 x float] %v127, float 0.0, 62
  %v129 = insertvalue [128 x float] %v128, float 0.0, 63
  %v130 = insertvalue [128 x float] %v129, float 0.0, 64
  %v131 = insertvalue [128 x float] %v130, float 0.0, 65
  %v132 = insertvalue [128 x float] %v131, float 0.0, 66
  %v133 = insertvalue [128 x float] %v132, float 0.0, 67
  %v134 = insertvalue [128 x float] %v133, float 0.0, 68
  %v135 = insertvalue [128 x float] %v134, float 0.0, 69
  %v136 = insertvalue [128 x float] %v135, float 0.0, 70
  %v137 = insertvalue [128 x float] %v136, float 0.0, 71
  %v138 = insertvalue [128 x float] %v137, float 0.0, 72
  %v139 = insertvalue [128 x float] %v138, float 0.0, 73
  %v140 = insertvalue [128 x float] %v139, float 0.0, 74
  %v141 = insertvalue [128 x float] %v140, float 0.0, 75
  %v142 = insertvalue [128 x float] %v141, float 0.0, 76
  %v143 = insertvalue [128 x float] %v142, float 0.0, 77
  %v144 = insertvalue [128 x float] %v143, float 0.0, 78
  %v145 = insertvalue [128 x float] %v144, float 0.0, 79
  %v146 = insertvalue [128 x float] %v145, float 0.0, 80
  %v147 = insertvalue [128 x float] %v146, float 0.0, 81
  %v148 = insertvalue [128 x float] %v147, float 0.0, 82
  %v149 = insertvalue [128 x float] %v148, float 0.0, 83
  %v150 = insertvalue [128 x float] %v149, float 0.0, 84
  %v151 = insertvalue [128 x float] %v150, float 0.0, 85
  %v152 = insertvalue [128 x float] %v151, float 0.0, 86
  %v153 = insertvalue [128 x float] %v152, float 0.0, 87
  %v154 = insertvalue [128 x float] %v153, float 0.0, 88
  %v155 = insertvalue [128 x float] %v154, float 0.0, 89
  %v156 = insertvalue [128 x float] %v155, float 0.0, 90
  %v157 = insertvalue [128 x float] %v156, float 0.0, 91
  %v158 = insertvalue [128 x float] %v157, float 0.0, 92
  %v159 = insertvalue [128 x float] %v158, float 0.0, 93
  %v160 = insertvalue [128 x float] %v159, float 0.0, 94
  %v161 = insertvalue [128 x float] %v160, float 0.0, 95
  %v162 = insertvalue [128 x float] %v161, float 0.0, 96
  %v163 = insertvalue [128 x float] %v162, float 0.0, 97
  %v164 = insertvalue [128 x float] %v163, float 0.0, 98
  %v165 = insertvalue [128 x float] %v164, float 0.0, 99
  %v166 = insertvalue [128 x float] %v165, float 0.0, 100
  %v167 = insertvalue [128 x float] %v166, float 0.0, 101
  %v168 = insertvalue [128 x float] %v167, float 0.0, 102
  %v169 = insertvalue [128 x float] %v168, float 0.0, 103
  %v170 = insertvalue [128 x float] %v169, float 0.0, 104
  %v171 = insertvalue [128 x float] %v170, float 0.0, 105
  %v172 = insertvalue [128 x float] %v171, float 0.0, 106
  %v173 = insertvalue [128 x float] %v172, float 0.0, 107
  %v174 = insertvalue [128 x float] %v173, float 0.0, 108
  %v175 = insertvalue [128 x float] %v174, float 0.0, 109
  %v176 = insertvalue [128 x float] %v175, float 0.0, 110
  %v177 = insertvalue [128 x float] %v176, float 0.0, 111
  %v178 = insertvalue [128 x float] %v177, float 0.0, 112
  %v179 = insertvalue [128 x float] %v178, float 0.0, 113
  %v180 = insertvalue [128 x float] %v179, float 0.0, 114
  %v181 = insertvalue [128 x float] %v180, float 0.0, 115
  %v182 = insertvalue [128 x float] %v181, float 0.0, 116
  %v183 = insertvalue [128 x float] %v182, float 0.0, 117
  %v184 = insertvalue [128 x float] %v183, float 0.0, 118
  %v185 = insertvalue [128 x float] %v184, float 0.0, 119
  %v186 = insertvalue [128 x float] %v185, float 0.0, 120
  %v187 = insertvalue [128 x float] %v186, float 0.0, 121
  %v188 = insertvalue [128 x float] %v187, float 0.0, 122
  %v189 = insertvalue [128 x float] %v188, float 0.0, 123
  %v190 = insertvalue [128 x float] %v189, float 0.0, 124
  %v191 = insertvalue [128 x float] %v190, float 0.0, 125
  %v192 = insertvalue [128 x float] %v191, float 0.0, 126
  %v193 = insertvalue [128 x float] %v192, float 0.0, 127
  store [128 x float] %v193, ptr %v30
  %v194 = load [128 x float], ptr %v30
  store [128 x float] %v194, ptr %v31
  br label %bb23
bb23:
  %v195 = phi i64 [ 0, %bb22 ], [ %v367, %bb27 ]
  %v196 = icmp ult i64 %v195, %v48
  %v197 = xor i1 %v196, 1
  br i1 %v197, label %bb89, label %bb88
bb24:
  unreachable
bb25:
  %v198 = extractvalue { i8, i64 } %v366, 1
  %v199 = icmp ult i64 %v198, 128
  br i1 %v199, label %bb27, label %bb104
bb26:
  br label %bb28
bb27:
  %v200 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v198
  store float 0.0, ptr %v200
  br label %bb23
bb28:
  %v201 = phi i64 [ 0, %bb26 ], [ %v269, %bb45 ]
  %v202 = icmp ult i64 %v201, %v32
  %v203 = xor i1 %v202, 1
  br i1 %v203, label %bb46, label %bb29
bb29:
  %v204 = urem i64 %v37, 64
  %v205 = add i64 %v201, %v204
  %v206 = icmp ult i64 %v37, %v48
  %v207 = xor i1 %v206, 1
  br i1 %v207, label %bb33, label %bb30
bb30:
  %v208 = icmp ult i64 %v205, %v32
  %v209 = xor i1 %v208, 1
  br i1 %v209, label %bb33, label %bb31
bb31:
  %v210 = mul i64 %v37, %v45
  %v211 = add i64 %v210, %v205
  %v212 = extractvalue { ptr, i64 } %v21, 1
  %v213 = icmp ult i64 %v211, %v212
  br i1 %v213, label %bb32, label %bb105
bb32:
  %v214 = extractvalue { ptr, i64 } %v21, 0
  %v215 = getelementptr inbounds i16, ptr %v214, i64 %v211
  %v216 = load i16, ptr %v215
  %v217 = zext i16 %v216 to i32
  %v218 = and i32 16, 31
  %v219 = shl i32 %v217, %v218
  %v220 = bitcast i32 %v219 to float
  br label %bb33
bb33:
  %v221 = add i64 %v201, %v204
  %v222 = udiv i64 %v37, 64
  %v223 = icmp ult i64 %v221, %v32
  %v224 = xor i1 %v223, 1
  br i1 %v224, label %bb36, label %bb34
bb34:
  %v225 = icmp ult i64 %v222, %v51
  %v226 = xor i1 %v225, 1
  br i1 %v226, label %bb36, label %bb35
bb35:
  %v227 = mul i64 %v221, %v45
  %v228 = add i64 %v227, %v222
  br label %bb36
bb36:
  br label %bb37
bb37:
  %v229 = phi i64 [ 0, %bb36 ], [ %v268, %bb44 ]
  %v230 = icmp ult i64 %v229, 64
  %v231 = xor i1 %v230, 1
  br i1 %v231, label %bb45, label %bb38
bb38:
  %v232 = urem i64 %v37, %v48
  %v233 = mul i64 %v232, %v45
  %v234 = add i64 %v233, %v201
  %v235 = add i64 %v234, %v229
  %v236 = add i64 %v201, %v229
  %v237 = mul i64 %v236, %v45
  %v238 = udiv i64 %v37, %v48
  %v239 = add i64 %v237, %v238
  %v240 = extractvalue { ptr, i64 } %v21, 1
  %v241 = icmp ult i64 %v235, %v240
  %v242 = xor i1 %v241, 1
  br i1 %v242, label %bb44, label %bb39
bb39:
  %v243 = extractvalue { ptr, i64 } %v22, 1
  %v244 = icmp ult i64 %v239, %v243
  %v245 = xor i1 %v244, 1
  br i1 %v245, label %bb44, label %bb40
bb40:
  %v246 = extractvalue { ptr, i64 } %v21, 0
  %v247 = getelementptr inbounds i16, ptr %v246, i64 %v235
  %v248 = load i16, ptr %v247
  %v249 = zext i16 %v248 to i32
  %v250 = and i32 16, 31
  %v251 = shl i32 %v249, %v250
  %v252 = bitcast i32 %v251 to float
  %v253 = extractvalue { ptr, i64 } %v22, 0
  %v254 = getelementptr inbounds i16, ptr %v253, i64 %v239
  %v255 = load i16, ptr %v254
  %v256 = zext i16 %v255 to i32
  %v257 = and i32 16, 31
  %v258 = shl i32 %v256, %v257
  %v259 = bitcast i32 %v258 to float
  %v260 = icmp ult i64 %v232, %v48
  %v261 = xor i1 %v260, 1
  br i1 %v261, label %bb43, label %bb41
bb41:
  %v262 = fmul float %v252, %v259
  %v263 = icmp ult i64 %v232, 128
  br i1 %v263, label %bb42, label %bb106
bb42:
  %v264 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v232
  %v265 = load float, ptr %v264
  %v266 = fadd float %v265, %v262
  %v267 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v232
  store float %v266, ptr %v267
  br label %bb43
bb43:
  br label %bb44
bb44:
  %v268 = add i64 %v229, 1
  br label %bb37
bb45:
  %v269 = add i64 %v201, 64
  br label %bb28
bb46:
  br label %bb47
bb47:
  %v270 = phi float [ 0xFFF0000000000000, %bb46 ], [ %v282, %bb53 ]
  %v271 = phi i64 [ 0, %bb46 ], [ %v377, %bb53 ]
  %v272 = icmp ult i64 %v271, %v48
  %v273 = xor i1 %v272, 1
  br i1 %v273, label %bb93, label %bb92
bb48:
  %v274 = extractvalue { i8, i64 } %v376, 1
  %v275 = icmp ult i64 %v274, 128
  br i1 %v275, label %bb50, label %bb107
bb49:
  br label %bb54
bb50:
  %v276 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v274
  %v277 = load float, ptr %v276
  %v278 = fcmp ogt float %v277, %v270
  %v279 = xor i1 %v278, 1
  br i1 %v279, label %bb52, label %bb51
bb51:
  %v280 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v274
  %v281 = load float, ptr %v280
  br label %bb53
bb52:
  br label %bb53
bb53:
  %v282 = phi float [ %v281, %bb51 ], [ %v270, %bb52 ]
  br label %bb47
bb54:
  %v283 = phi float [ 0.0, %bb49 ], [ %v296, %bb58 ]
  %v284 = phi i64 [ 0, %bb49 ], [ %v387, %bb58 ]
  %v285 = icmp ult i64 %v284, %v48
  %v286 = xor i1 %v285, 1
  br i1 %v286, label %bb97, label %bb96
bb55:
  %v287 = extractvalue { i8, i64 } %v386, 1
  %v288 = icmp ult i64 %v287, 128
  br i1 %v288, label %bb57, label %bb108
bb56:
  %v289 = fcmp ogt float %v283, 0.0
  %v290 = xor i1 %v289, 1
  br i1 %v290, label %bb60, label %bb59
bb57:
  %v291 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v287
  %v292 = load float, ptr %v291
  %v293 = fsub float %v292, %v270
  %v294 = call float @tcgen05_attention__exp_approx(float %v293)
  br label %bb58
bb58:
  %v295 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v287
  store float %v294, ptr %v295
  %v296 = fadd float %v283, %v294
  br label %bb54
bb59:
  br label %bb61
bb60:
  br label %bb65
bb61:
  %v297 = phi i64 [ 0, %bb59 ], [ %v397, %bb64 ]
  %v298 = icmp ult i64 %v297, %v48
  %v299 = xor i1 %v298, 1
  br i1 %v299, label %bb101, label %bb100
bb62:
  %v300 = extractvalue { i8, i64 } %v396, 1
  %v301 = icmp ult i64 %v300, 128
  br i1 %v301, label %bb64, label %bb109
bb63:
  br label %bb65
bb64:
  %v302 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v300
  %v303 = load float, ptr %v302
  %v304 = fdiv float %v303, %v283
  %v305 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v300
  store float %v304, ptr %v305
  br label %bb61
bb65:
  br label %bb66
bb66:
  %v306 = phi i64 [ 0, %bb65 ], [ %v343, %bb75 ]
  %v307 = icmp ult i64 %v306, %v51
  %v308 = xor i1 %v307, 1
  br i1 %v308, label %bb76, label %bb67
bb67:
  %v309 = urem i64 %v37, 64
  %v310 = add i64 %v306, %v309
  %v311 = udiv i64 %v37, 64
  %v312 = icmp ult i64 %v310, %v51
  %v313 = xor i1 %v312, 1
  br i1 %v313, label %bb75, label %bb68
bb68:
  %v314 = icmp ult i64 %v311, %v32
  %v315 = xor i1 %v314, 1
  br i1 %v315, label %bb75, label %bb69
bb69:
  %v316 = mul i64 %v310, %v45
  %v317 = add i64 %v316, %v311
  %v318 = extractvalue { ptr, i64 } %v23, 1
  %v319 = icmp ult i64 %v317, %v318
  %v320 = xor i1 %v319, 1
  br i1 %v320, label %bb75, label %bb70
bb70:
  %v321 = urem i64 %v310, %v48
  %v322 = icmp ult i64 %v321, 128
  br i1 %v322, label %bb71, label %bb110
bb71:
  %v323 = getelementptr inbounds [128 x float], ptr %v31, i32 0, i64 %v321
  %v324 = load float, ptr %v323
  %v325 = extractvalue { ptr, i64 } %v23, 0
  %v326 = getelementptr inbounds i16, ptr %v325, i64 %v317
  %v327 = load i16, ptr %v326
  %v328 = zext i16 %v327 to i32
  %v329 = and i32 16, 31
  %v330 = shl i32 %v328, %v329
  %v331 = bitcast i32 %v330 to float
  %v332 = urem i64 %v37, %v48
  %v333 = icmp ult i64 %v332, %v48
  %v334 = xor i1 %v333, 1
  br i1 %v334, label %bb74, label %bb72
bb72:
  %v335 = fmul float %v324, %v331
  %v336 = mul i64 %v332, %v32
  %v337 = add i64 %v336, %v311
  %v338 = icmp ult i64 %v337, 128
  br i1 %v338, label %bb73, label %bb111
bb73:
  %v339 = getelementptr inbounds [128 x float], ptr %v30, i32 0, i64 %v337
  %v340 = load float, ptr %v339
  %v341 = fadd float %v340, %v335
  %v342 = getelementptr inbounds [128 x float], ptr %v30, i32 0, i64 %v337
  store float %v341, ptr %v342
  br label %bb74
bb74:
  br label %bb75
bb75:
  %v343 = add i64 %v306, 64
  br label %bb66
bb76:
  %v344 = urem i64 %v37, %v48
  %v345 = icmp ult i64 %v37, %v48
  %v346 = xor i1 %v345, 1
  br i1 %v346, label %bb82, label %bb77
bb77:
  %v347 = mul i64 %v344, %v45
  %v348 = add i64 %v347, %v44
  %v349 = extractvalue { ptr, i64 } %v24, 0
  %v350 = extractvalue { ptr, i64 } %v24, 1
  %v351 = icmp ult i64 %v348, %v350
  %v352 = xor i1 %v351, 1
  br i1 %v352, label %bb80, label %bb78
bb78:
  %v353 = icmp ult i64 %v344, 128
  br i1 %v353, label %bb79, label %bb112
bb79:
  %v354 = getelementptr inbounds [128 x float], ptr %v30, i32 0, i64 %v344
  %v355 = load float, ptr %v354
  %v356 = getelementptr inbounds float, ptr %v349, i64 %v348
  store float %v355, ptr %v356
  br label %bb81
bb80:
  br label %bb81
bb81:
  br label %bb82
bb82:
  %v357 = xor i1 %v359, 1
  br i1 %v357, label %bb84, label %bb83
bb83:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v60, i32 512) #0
  br label %bb84
bb84:
  br label %bb86
bb85:
  br label %bb86
bb86:
  ret void
bb87:
  %v358 = udiv i32 %v38, 32
  %v359 = icmp eq i32 %v358, 0
  %v360 = icmp eq i64 %v37, 0
  %v361 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  br label %bb2
bb88:
  %v362 = add i64 %v195, 1
  %v363 = insertvalue { i8, i64 } undef, i8 1, 0
  %v364 = insertvalue { i8, i64 } %v363, i64 %v195, 1
  br label %bb90
bb89:
  %v365 = insertvalue { i8, i64 } undef, i8 0, 0
  br label %bb90
bb90:
  %v366 = phi { i8, i64 } [ %v364, %bb88 ], [ %v365, %bb89 ]
  %v367 = phi i64 [ %v362, %bb88 ], [ %v195, %bb89 ]
  %v368 = extractvalue { i8, i64 } %v366, 0
  %v369 = zext i8 %v368 to i64
  %v370 = icmp eq i64 %v369, 0
  br i1 %v370, label %bb26, label %bb91
bb91:
  %v371 = icmp eq i64 %v369, 1
  br i1 %v371, label %bb25, label %bb24
bb92:
  %v372 = add i64 %v271, 1
  %v373 = insertvalue { i8, i64 } undef, i8 1, 0
  %v374 = insertvalue { i8, i64 } %v373, i64 %v271, 1
  br label %bb94
bb93:
  %v375 = insertvalue { i8, i64 } undef, i8 0, 0
  br label %bb94
bb94:
  %v376 = phi { i8, i64 } [ %v374, %bb92 ], [ %v375, %bb93 ]
  %v377 = phi i64 [ %v372, %bb92 ], [ %v271, %bb93 ]
  %v378 = extractvalue { i8, i64 } %v376, 0
  %v379 = zext i8 %v378 to i64
  %v380 = icmp eq i64 %v379, 0
  br i1 %v380, label %bb49, label %bb95
bb95:
  %v381 = icmp eq i64 %v379, 1
  br i1 %v381, label %bb48, label %bb24
bb96:
  %v382 = add i64 %v284, 1
  %v383 = insertvalue { i8, i64 } undef, i8 1, 0
  %v384 = insertvalue { i8, i64 } %v383, i64 %v284, 1
  br label %bb98
bb97:
  %v385 = insertvalue { i8, i64 } undef, i8 0, 0
  br label %bb98
bb98:
  %v386 = phi { i8, i64 } [ %v384, %bb96 ], [ %v385, %bb97 ]
  %v387 = phi i64 [ %v382, %bb96 ], [ %v284, %bb97 ]
  %v388 = extractvalue { i8, i64 } %v386, 0
  %v389 = zext i8 %v388 to i64
  %v390 = icmp eq i64 %v389, 0
  br i1 %v390, label %bb56, label %bb99
bb99:
  %v391 = icmp eq i64 %v389, 1
  br i1 %v391, label %bb55, label %bb24
bb100:
  %v392 = add i64 %v297, 1
  %v393 = insertvalue { i8, i64 } undef, i8 1, 0
  %v394 = insertvalue { i8, i64 } %v393, i64 %v297, 1
  br label %bb102
bb101:
  %v395 = insertvalue { i8, i64 } undef, i8 0, 0
  br label %bb102
bb102:
  %v396 = phi { i8, i64 } [ %v394, %bb100 ], [ %v395, %bb101 ]
  %v397 = phi i64 [ %v392, %bb100 ], [ %v297, %bb101 ]
  %v398 = extractvalue { i8, i64 } %v396, 0
  %v399 = zext i8 %v398 to i64
  %v400 = icmp eq i64 %v399, 0
  br i1 %v400, label %bb63, label %bb103
bb103:
  %v401 = icmp eq i64 %v399, 1
  br i1 %v401, label %bb62, label %bb24
bb104:
  unreachable
bb105:
  unreachable
bb106:
  unreachable
bb107:
  unreachable
bb108:
  unreachable
bb109:
  unreachable
bb110:
  unreachable
bb111:
  unreachable
bb112:
  unreachable
}

define float @tcgen05_attention__exp_approx(float %v0) {
entry:
  br label %bb0
bb0:
  %v1 = phi float [ %v0, %entry ]
  %v2 = fneg float %v1
  %v3 = fcmp ogt float %v2, 88.0
  %v4 = xor i1 %v3, 1
  br i1 %v4, label %bb2, label %bb1
bb1:
  br label %bb5
bb2:
  %v5 = fcmp olt float %v2, 0.0000009999999974752427
  %v6 = xor i1 %v5, 1
  br i1 %v6, label %bb4, label %bb3
bb3:
  br label %bb5
bb4:
  %v7 = fmul float %v2, %v2
  %v8 = fmul float %v7, %v2
  %v9 = fmul float %v8, %v2
  %v10 = fmul float %v9, %v2
  %v11 = fadd float 1.0, %v2
  %v12 = fmul float %v7, 0.5
  %v13 = fadd float %v11, %v12
  %v14 = fmul float %v8, 0.1666666716337204
  %v15 = fadd float %v13, %v14
  %v16 = fmul float %v9, 0.0416666679084301
  %v17 = fadd float %v15, %v16
  %v18 = fmul float %v10, 0.00833333283662796
  %v19 = fadd float %v17, %v18
  %v20 = fdiv float 1.0, %v19
  br label %bb5
bb5:
  %v21 = phi float [ 0.0, %bb1 ], [ 1.0, %bb3 ], [ %v20, %bb4 ]
  ret float %v21
}


attributes #0 = { convergent }
