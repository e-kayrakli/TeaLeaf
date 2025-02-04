module field_summary {
    use settings;
    use chunks;
    use IO;
    use profile;
    use GPU;

   /*
    * 		FIELD SUMMARY KERNEL
    * 		Calculates aggregates of values in field.
    */	
    // The field summary kernel
    proc field_summary (const ref halo_depth: int(32), const ref volume: [?Domain] real,
                        const ref density: [Domain] real, const ref energy0: [Domain] real, 
                        const ref u: [Domain] real, ref vol: real, ref mass: real, 
                        ref ie: real, ref temp: real,
                        const ref reduced_local_domain: subdomain(Domain), const ref reduced_OneD: domain(1,int(32))) {
        startProfiling("field_summary");

        if useGPU {
            var tempVol: [reduced_local_domain] real = noinit, 
            tempMass: [reduced_local_domain] real = noinit,
            tempIe: [reduced_local_domain] real = noinit,
            tempTemp: [reduced_local_domain] real = noinit;

            forall oneDIdx in reduced_OneD {
                const ij = reduced_local_domain.orderToIndex(oneDIdx);
                var cellMass: real;
                tempVol[ij] = volume[ij];
                cellMass = volume[ij] * density[ij];
                tempMass[ij] = cellMass;
                tempIe[ij] = cellMass * energy0[ij];
                tempTemp[ij] = cellMass * u[ij];
            }

            vol = gpuSumReduce(tempVol);
            mass = gpuSumReduce(tempMass);
            ie = gpuSumReduce(tempIe);
            temp = gpuSumReduce(tempTemp);
        } else {
            var localVol = 0.0,
            localMass = 0.0,
            localIe = 0.0,
            localTemp = 0.0;
        
            forall ij in reduced_local_domain with (+ reduce localVol, + reduce localMass, + reduce localIe, + reduce localTemp) {
                var cellMass: real;
                localVol += volume[ij];
                cellMass = volume[ij] * density[ij];
                localMass += cellMass;
                localIe += cellMass * energy0[ij];
                localTemp += cellMass * u[ij];
            }
            
            vol = localVol;
            mass = localMass;
            ie = localIe;
            temp = localTemp;
        }
        stopProfiling("field_summary");
    }

/*
 * 		FIELD SUMMARY DRIVER
 */	

    // Invokes the set chunk data kernel
    proc field_summary_driver(ref chunk_var :chunks.Chunk, ref setting_var : settings.setting,
        const in is_solve_finished: bool) {
        
        var vol, ie, temp, mass : real;

        field_summary(setting_var.halo_depth, chunk_var.volume, 
        chunk_var.density, chunk_var.energy0, chunk_var.u, vol, mass, ie, temp, chunk_var.reduced_local_domain, chunk_var.reduced_OneD);

        if (setting_var.check_result && is_solve_finished) { 
            var checking_value : real;
            get_checking_value(setting_var, checking_value);

            writeln("Expected: \n", checking_value);
            writeln("Actual: \n", temp);

            var qa_diff: real = abs(100.0*(temp/checking_value)-100.0);

            if qa_diff < 0.001 then writeln("Passed with qa_diff of ", qa_diff);
            else writeln("Failed with qa_diff of ", qa_diff);
        } 
    }

    // Fetches the checking value from the test problems file
    proc get_checking_value (const ref setting_var : settings.setting, ref checking_value : real) {
        var counter : int;
        try {
            var tea_prob = open (setting_var.test_problem_filename, ioMode.r);
            var tea_prob_reader = tea_prob.reader();
            
            var x : int;
            var y : int;
            var num_steps: int;
            var line : string;
            tea_prob_reader.read(x, y, num_steps, checking_value);

            for line in tea_prob_reader.lines() {
                counter += 1;
                if (x == setting_var.grid_x_cells && y == setting_var.grid_y_cells && 
                    num_steps == setting_var.end_step) {
                    // Found the problem in the file
                    tea_prob.close();
                    return;
                }
                tea_prob_reader.read(x, y, num_steps, checking_value);
            }
            tea_prob.close();
        } catch {
            writeln("Error parsing at line: ", counter);
        }
    }
}   
