; ModuleID = 'builtin.module'
source_filename = "tcgen05_gemm"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_2 = addrspace(3) global [8192 x float] zeroinitializer, align 128
@__shared_mem_1 = addrspace(3) global [8192 x float] zeroinitializer, align 128
@__shared_mem_0 = addrspace(3) global [1 x i32] zeroinitializer, align 4
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare void @llvm.nvvm.barrier0() #0

define ptx_kernel void @tcgen05_gemm(i32 %v0, i32 %v1, i32 %v2, float %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, float %v8, ptr %v9, i64 %v10) {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v7, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v9, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v10, 1
  br label %bb0
bb0:
  %v17 = phi i32 [ %v0, %entry ]
  %v18 = phi i32 [ %v1, %entry ]
  %v19 = phi i32 [ %v2, %entry ]
  %v20 = phi float [ %v3, %entry ]
  %v21 = phi { ptr, i64 } [ %v12, %entry ]
  %v22 = phi { ptr, i64 } [ %v14, %entry ]
  %v23 = phi float [ %v8, %entry ]
  %v24 = phi { ptr, i64 } [ %v16, %entry ]
  %v25 = zext i32 %v17 to i64
  %v26 = zext i32 %v18 to i64
  %v27 = zext i32 %v19 to i64
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb1
bb1:
  %v29 = zext i32 %v28 to i64
  %v30 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  br label %bb28
bb2:
  %v31 = zext i32 %v112 to i64
  %v32 = mul i64 %v31, 128
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  br label %bb3
bb3:
  %v34 = zext i32 %v33 to i64
  %v35 = mul i64 %v34, 128
  %v36 = urem i64 %v29, 128
  %v37 = add i64 %v32, %v36
  %v38 = udiv i64 %v29, 128
  %v39 = add i64 %v35, %v38
  %v40 = icmp ult i64 %v37, %v25
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb27, label %bb4
bb4:
  %v42 = icmp ult i64 %v39, %v26
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb27, label %bb5
bb5:
  %v44 = xor i1 %v111, 1
  br i1 %v44, label %bb8, label %bb6
bb6:
  %v45 = addrspacecast ptr addrspace(3) @__shared_mem_0 to ptr
  call void asm sideeffect "{ .reg .u64 %shared64; .reg .u32 %shared32; cvta.to.shared.u64 %shared64, $0; cvt.u32.u64 %shared32, %shared64; tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%shared32], $1; }", "l,r,~{memory}"(ptr %v45, i32 512) #0
  br label %bb7
bb7:
  br label %bb8
bb8:
  call void @llvm.nvvm.barrier0() #0
  br label %bb9
bb9:
  %v47 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v48 = addrspacecast ptr addrspace(3) %v47 to ptr
  %v49 = load i32, ptr %v48
  br label %bb10
bb10:
  %v50 = phi float [ 0.0, %bb9 ], [ %v82, %bb23 ]
  %v51 = phi i64 [ 0, %bb9 ], [ %v99, %bb23 ]
  %v52 = icmp ult i64 %v51, %v27
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb24, label %bb11
bb11:
  %v54 = udiv i64 %v29, 128
  %v55 = add i64 %v51, %v54
  %v56 = icmp ult i64 %v55, %v27
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb15, label %bb12
bb12:
  %v58 = mul i64 %v37, %v27
  %v59 = add i64 %v58, %v55
  %v60 = extractvalue { ptr, i64 } %v21, 1
  %v61 = icmp ult i64 %v59, %v60
  br i1 %v61, label %bb13, label %bb29
bb13:
  %v62 = extractvalue { ptr, i64 } %v21, 0
  %v63 = getelementptr inbounds float, ptr %v62, i64 %v59
  %v64 = load float, ptr %v63
  %v65 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v29
  br label %bb14
bb14:
  store float %v64, ptr addrspace(3) %v65
  br label %bb15
bb15:
  %v66 = add i64 %v51, %v36
  %v67 = icmp ult i64 %v66, %v27
  %v68 = xor i1 %v67, 1
  br i1 %v68, label %bb19, label %bb16
bb16:
  %v69 = mul i64 %v66, %v26
  %v70 = add i64 %v69, %v39
  %v71 = extractvalue { ptr, i64 } %v22, 1
  %v72 = icmp ult i64 %v70, %v71
  br i1 %v72, label %bb17, label %bb30
bb17:
  %v73 = extractvalue { ptr, i64 } %v22, 0
  %v74 = getelementptr inbounds float, ptr %v73, i64 %v70
  %v75 = load float, ptr %v74
  %v76 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_2, i64 %v29
  br label %bb18
bb18:
  store float %v75, ptr addrspace(3) %v76
  br label %bb19
bb19:
  call void @llvm.nvvm.barrier0() #0
  br label %bb20
bb20:
  %v78 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v79 = addrspacecast ptr addrspace(3) %v78 to ptr
  %v80 = bitcast ptr addrspace(3) @__shared_mem_2 to ptr addrspace(3)
  %v81 = addrspacecast ptr addrspace(3) %v80 to ptr
  br label %bb21
bb21:
  %v82 = phi float [ %v50, %bb20 ], [ %v97, %bb22 ]
  %v83 = phi i64 [ 0, %bb20 ], [ %v98, %bb22 ]
  %v84 = icmp ult i64 %v83, 64
  %v85 = xor i1 %v84, 1
  br i1 %v85, label %bb23, label %bb22
bb22:
  %v86 = sub i64 %v37, %v32
  %v87 = mul i64 %v86, 64
  %v88 = add i64 %v87, %v83
  %v89 = getelementptr inbounds float, ptr %v79, i64 %v88
  %v90 = load float, ptr %v89
  %v91 = mul i64 %v83, 128
  %v92 = sub i64 %v39, %v35
  %v93 = add i64 %v91, %v92
  %v94 = getelementptr inbounds float, ptr %v81, i64 %v93
  %v95 = load float, ptr %v94
  %v96 = fmul float %v90, %v95
  %v97 = fadd float %v82, %v96
  %v98 = add i64 %v83, 1
  br label %bb21
bb23:
  %v99 = add i64 %v51, 64
  br label %bb10
bb24:
  %v100 = mul i64 %v37, %v26
  %v101 = add i64 %v100, %v39
  %v102 = extractvalue { ptr, i64 } %v24, 0
  %v103 = fmul float %v20, %v50
  %v104 = getelementptr inbounds float, ptr %v102, i64 %v101
  %v105 = load float, ptr %v104
  %v106 = fmul float %v23, %v105
  %v107 = fadd float %v103, %v106
  store float %v107, ptr %v104
  %v108 = xor i1 %v111, 1
  br i1 %v108, label %bb26, label %bb25
bb25:
  call void asm sideeffect "tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, $1;", "r,r,~{memory}"(i32 %v49, i32 512) #0
  br label %bb26
bb26:
  br label %bb27
bb27:
  ret void
bb28:
  %v109 = udiv i32 %v30, 32
  %v110 = icmp eq i64 %v29, 0
  %v111 = icmp eq i32 %v109, 0
  %v112 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
  br label %bb2
bb29:
  unreachable
bb30:
  unreachable
}


attributes #0 = { convergent }
