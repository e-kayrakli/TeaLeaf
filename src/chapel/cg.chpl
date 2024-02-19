/*
 *		CONJUGATE GRADIENT SOLVER KERNEL
 */
module cg {
    use settings;
    use Math;
    use profile;
    use chunks;
    use GPU;

    proc cg_init(const in x: int(32), const in y: int(32), const in halo_depth: int(32), const in coefficient: int, const in rx: real, 
                const in ry: real, out rro: real, const ref density: [?Domain] real, const ref energy: [Domain] real,
                ref u: [Domain] real,  ref p: [Domain] real,  ref r: [Domain] real,  ref w: [Domain] real,  
                ref kx: [Domain] real, ref ky: [Domain] real, ref temp: [Domain] real,
                const ref reduced_local_domain: subdomain(Domain), const ref reduced_OneD: domain(1,int(32)), const ref local_domain: subdomain(Domain), const ref OneD: domain(1,int(32))) {

        if coefficient != CONDUCTIVITY && coefficient != RECIP_CONDUCTIVITY {
            writeln("Coefficient ", coefficient, " is not valid.\n");
            exit(-1);
        }

        startProfiling("cg_init");

        //foreach ij in Domain {
        forall oneDIdx in OneD {
            const ij = local_domain.orderToIndex(oneDIdx);
            p[ij] = 0;
            r[ij] = 0;
            u[ij] = energy[ij] *density[ij];
            w[ij] = if (coefficient == CONDUCTIVITY) then density[ij] else 1.0/density[ij];
        }
        const inner_1 = Domain[halo_depth..<y-1, halo_depth..<x-1];
        //forall (i, j) in inner_1 {
        forall oneDIdx in 0..#inner_1.size {
            const (i,j) = inner_1.orderToIndex(oneDIdx);
            kx[i, j] = rx*(w[i-1, j]+w[i, j]) / (2.0*w[i-1, j]*w[i, j]);
            ky[i, j] = ry*(w[i, j-1]+w[i, j]) / (2.0*w[i, j-1]*w[i, j]);
        }   
         
        if useGPU then {  // GPU version of Loop
            //forall (i, j) in Domain.expand(-halo_depth) {
            forall oneDIdx in reduced_OneD {
                const (i,j) = reduced_local_domain.orderToIndex(oneDIdx);
                const smvp = (1.0 + (kx[i+1, j]+kx[i, j])
                    + (ky[i, j+1]+ky[i, j]))*u[i, j]
                    - (kx[i+1, j]*u[i+1, j]+kx[i, j]*u[i-1, j])
                    - (ky[i, j+1]*u[i, j+1]+ky[i, j]*u[i, j-1]);
                w[i, j] = smvp;
                r[i,j] = u[i,j] - smvp;
                p[i,j] = r[i,j];
                
                temp[i, j] = p[i,j]*p[i,j]; 
            }   
            rro = gpuSumReduce(temp);
        } else { // CPU version
            var rro_temp : real;
            forall (i, j) in reduced_local_domain with (+ reduce rro_temp) {
                const smvp = (1.0 + (kx[i+1, j]+kx[i, j])
                    + (ky[i, j+1]+ky[i, j]))*u[i, j]
                    - (kx[i+1, j]*u[i+1, j]+kx[i, j]*u[i-1, j])
                    - (ky[i, j+1]*u[i, j+1]+ky[i, j]*u[i, j-1]);
                w[i, j] = smvp;
                r[i,j] = u[i,j] - smvp;
                p[i,j] = r[i,j];
                
                rro_temp += p[i,j]*p[i,j];
            }   
            rro += rro_temp;
        }

        stopProfiling("cg_init");
    }

    // Calculates w
    proc cg_calc_w (const in halo_depth: int(32), out pw: real, const ref p: [?Domain] real, 
                    ref w: [Domain] real, const ref kx: [Domain] real, const ref ky: [Domain] real, ref temp: [Domain] real,
                    const ref reduced_local_domain: subdomain(Domain), const ref reduced_OneD: domain(1,int(32)), const ref local_domain: subdomain(Domain), const ref OneD: domain(1,int(32))) {

        startProfiling("cg_calc_w");
        
        if useGPU {
            //forall (i, j) in Domain.expand(-halo_depth) {
            forall oneDIdx in reduced_OneD {
                const (i,j) = reduced_local_domain.orderToIndex(oneDIdx);
                const smvp = (1.0 + (kx[i+1, j]+kx[i, j])
                    + (ky[i, j+1]+ky[i, j]))*p[i, j]
                    - (kx[i+1, j]*p[i+1, j]+kx[i, j]*p[i-1, j])
                    - (ky[i, j+1]*p[i, j+1]+ky[i, j]*p[i, j-1]);
                w[i,j] = smvp;
                temp[i, j] = smvp * p[i, j]; 
            }
            pw = gpuSumReduce(temp);
        } else {
            var pw_temp : real;
            forall (i, j) in reduced_local_domain with (+ reduce pw_temp) {
                const smvp = (1.0 + (kx[i+1, j]+kx[i, j])
                    + (ky[i, j+1]+ky[i, j]))*p[i, j]
                    - (kx[i+1, j]*p[i+1, j]+kx[i, j]*p[i-1, j])
                    - (ky[i, j+1]*p[i, j+1]+ky[i, j]*p[i, j-1]);
                w[i,j] = smvp;
                
                pw_temp += smvp * p[i, j]; 
            }   
    
            pw += pw_temp;
        }
        stopProfiling("cg_calc_w");
    }
    
    // Calculates u and r
    proc cg_calc_ur(const in halo_depth: int(32), const in alpha: real, out rrn: real, 
                    ref u: [?Domain] real, const ref p: [Domain] real, 
                    ref r: [Domain] real, const ref w: [Domain] real, ref temp: [Domain] real,
                    const ref reduced_local_domain: subdomain(Domain), const ref reduced_OneD: domain(1,int(32)), const ref local_domain: subdomain(Domain), const ref OneD: domain(1,int(32))) {
        startProfiling("cg_calc_ur");

        if useGPU {
            //forall (i, j) in Domain.expand(-halo_depth) {
            startProfiling("cg_calc_ur_loop");
            forall oneDIdx in reduced_OneD {
                const ij = reduced_local_domain.orderToIndex(oneDIdx);
                u[ij] += alpha * p[ij];
                r[ij] -= alpha * w[ij];
                temp[ij] = r[ij] ** 2;
            }
            stopProfiling("cg_calc_ur_loop");

            startProfiling("cg_calc_ur_red");
            rrn = gpuSumReduce(temp);
            stopProfiling("cg_calc_ur_red");
        } else {
            var rrn_temp : real;
            forall ij in reduced_local_domain with (+ reduce rrn_temp) {
                u[ij] += alpha * p[ij];
                r[ij] -= alpha * w[ij];
                rrn_temp += r[ij] ** 2;
            }
            rrn += rrn_temp;
        }
        stopProfiling("cg_calc_ur");
    }

    // Calculates p
    proc cg_calc_p (const in halo_depth: int(32), const in beta: real, ref p: [?Domain] real, 
                    const ref r: [Domain] real,
                    const ref reduced_local_domain: subdomain(Domain), const ref reduced_OneD: domain(1,int(32)), const ref local_domain: subdomain(Domain), const ref OneD: domain(1,int(32))) {
        startProfiling("cg_calc_p");
        
        if useGPU {
            forall oneDIdx in reduced_OneD {
                const ij = reduced_local_domain.orderToIndex(oneDIdx);
                p[ij] = beta * p[ij] + r[ij];
            }
        } else {
            [ij in reduced_local_domain] p[ij] = beta * p[ij] + r[ij];
        }

        stopProfiling("cg_calc_p");
    }

}
