// autonomous.c


struct pos_param_block__f {
    double param1_f;
    double param2_f;
    double param3_f;
    double param4_f;
    double param5_f;
};


struct ret_s {
    pos_param_block__f *ret_params;
    void *module_context;
    void *label;
};


struct pos_param_block__f_with_return {
    struct ret_s ret_info;
    struct pos_param_block_f params;
};


struct init_sub {
    struct ret_s ret_info;
};


struct start_sub {
    struct ret_s ret_info;
};


struct getPosition_fn {
    struct ret_s ret_info;
    double start;
    double position;
};


struct move_sub {
    struct ret_s ret_info;
    double dist;
    double start;
    unsigned long dlt_vars;
};


struct {
    void *init_start_labels[];
    void *start_start_labels[];
    void *getPosition_start_labels[];
    void *move_start_labels[];
} autonomous_labels = {
    {&&init_start0};
    {&&start_start0};
    {&&getPosition_start0, &&getPosition_start1};
    {&&too_few_arguments, &&move_start1};
};


struct autonomous_module {
    struct init_sub       init_vars;
    struct start_sub      start_vars;
    struct getPosition_fn getPosition_vars;
    struct move_sub       move_vars;
};


struct top_modules {
    struct autonomuos_module autonomous;
    ...
};


int
main(int argc, char **argv, char **env) {
  void *current_module;


  init_start0:
    // set leftMotor.reversed? to: true?
    top_modules.leftMotor.reversed_ = 1;

    // return
    current_module =
      ((autonomous_module *)current_module)->init_vars.ret_info.module_context;
    goto *((autonomous_module *)current_module)->init_vars.ret_info.label;


  start_start0:
    // move 5ft
    ((autonomous_module *)current_module)->move_vars.dist = 60;
    ((autonomous_module *)current_module)->move_vars.ret_info.module_context
      = current_module;
    ((autonomous_module *)current_module)->move_vars.ret_info.label
      = &&start_ret;

    // call move
    current_module = ((autonomous_module *)current_module;
    goto *move_start_labels[1];
  
  start_ret:
    // return
    current_module =
      ((autonomous_module *)current_module)->start_vars.ret_info.module_context;
    goto *((autonomous_module *)current_module)->start_vars.ret_info.label;


  getPosition_start0:
    // start = 0
    ((autonomous_module *)current_module)->getPosition_vars.start = 0;

  getPosition_start1:
    // set position to:
    // -> (leftMotor.position + rightMotor.position) / 2 * 66.7 - start
    ((autonomous_module *)current_module)->getPosition_vars.position = 
      (leftMotor.position + rightMotor.position) / (2 * 66.7) - 
      ((autonomous_module *)current_module)->getPosition_vars.start;

    // telemetry.report key: "position" value: position
    top_modules.telemetry.report.kw_position.param1_s = "position";
    top_modules.telemetry.report.kw_value.param1_f = 
      ((autonomous_module *)current_module)->getPosition_vars.position;
    top_modules.telemetry.report.ret_info.module_context = current_module;
    top_modules.telemetry.report.ret_info.label = &&getPosition_ret;

    // call telemetry.report
    current_module = &top_modules.telemetry;
    goto *telemetry_labels.report_start_labels[1];

  getPosition_ret:
    // return position
    ((autonomous_module *)current_module)->getPosition_vars.ret_info
                                          .ret_params->param1_f =
      ((autonomous_module *)current_module)->getPosition_vars.position;
    current_module =
      ((autonomous_module *)current_module)->getPosition_vars.ret_info
                                            .module_context;
    goto *((autonomous_module *)current_module)->getPosition_vars
                                                .ret_info.label;

  move_start1;
    // set start to: {getPosition}
    ((autonomous_module *)current_module)->getPosition_vars.ret_info
                                          .ret_params
      = &((autonomous_module *)current_module)->move_vars.start;

    ((autonomous_module *)current_module)->getPosition_vars.ret_info
                                          .module_context = current_module;
    ((autonomous_module *)current_module)->getPosition_vars.ret_info.label
      = &&move_ret1;

    // call getPosition
    goto *autonomous_labels.getPosition_start_labels[0];

  move_ret1:
    // dlt conditions
    ((autonomous_module *)current_module)->move_vars.dlt_mask = 0;
    if (((autonomous_module *)current_module)->move_vars.dist >= 0) {
        ((autonomous_module *)current_module)->move_vars.dlt_mask |= 1;
    }

    // dlt actions
    switch (((autonomous_module *)current_module)->move_vars.dlt_mask) {
    case 1:
        goto move_forward;
    case 0:
        goto move_backward;
    }

  move_forward:
    // dlt conditions
    ((autonomous_module *)current_module)->move_vars.dlt_mask = 0;
    if (isActive) {
        ((autonomous_module *)current_module)->move_vars.dlt_mask |= 2;
    }
    if (((autonomous_module *)current_module)->move_vars.tmp1.float < 
        ((autonomous_module *)current_module)->move_vars.dist
    ) {
        ((autonomous_module *)current_module)->move_vars.dlt_mask |= 1;
    }

    // dlt actions
    switch (((autonomous_module *)current_module)->move_vars.dlt_mask) {
    case 3:
        // dual.set_power leftMotor: 100% rightMotor: 100%
        top_modules.dual.set_power.kw_leftMotor.param1_f = 1.0;
        top_modules.dual.set_power.kw_rightMotor.param1_f = 1.0;
        top_modules.dual.set_power.ret_info.module_context = current_module;
        top_modules.dual.set_power.ret_info.label = &&move_ret2;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual_labels.set_power_start_labels[0];

    move_ret2:
        // goto cycle.next returning_to: forward
        top_modules.cycle.set_power.ret_info.module_context = current_module;
        top_modules.cycle.set_power.ret_info.label = &&move_forward;

        // call dual.set_power
        current_module = &top_modules.cycle;
        goto *cycle_labels.next_start_labels[0];

    case 0:
    case 1:
    case 2:
        // dual.set_power leftMotor: 0 rightMotor: 0
        top_modules.dual.set_power.kw_leftMotor.param1_f = 0;
        top_modules.dual.set_power.kw_rightMotor.param1_f = 0;
        top_modules.dual.set_power.ret_info.module_context = current_module;
        top_modules.dual.set_power.ret_info.label = &&move_ret3;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual_labels.set_power_start_labels[0];

    move_ret3:
        // return
        current_module =
          ((autonomous_module *)current_module)->move_vars.ret_info
                                                .module_context;
        goto *((autonomous_module *)current_module)->move_vars.ret_info.label;
    }

  move_backward:
    // dlt conditions
    ((autonomous_module *)current_module)->move_vars.dlt_mask = 0;
    if (isActive) {
        ((autonomous_module *)current_module)->move_vars.dlt_mask |= 2;
    }
    if (((autonomous_module *)current_module)->move_vars.tmp1.float >
        ((autonomous_module *)current_module)->move_vars.dist
    ) {
        ((autonomous_module *)current_module)->move_vars.dlt_mask |= 1;
    }

    // dlt actions
    switch (((autonomous_module *)current_module)->move_vars.dlt_mask) {
    case 3:
        // dual.set_power leftMotor: -100% rightMotor: -100%
        top_modules.dual.set_power.kw_leftMotor.param1_f = -1.0;
        top_modules.dual.set_power.kw_rightMotor.param1_f = -1.0;
        top_modules.dual.set_power.ret_info.module_context = current_module;
        top_modules.dual.set_power.ret_info.label = &&move_ret4;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual_labels.set_power_start_labels[0];

    move_ret4:
        // goto cycle.next returning_to: forward
        top_modules.cycle.set_power.ret_info.module_context = current_module;
        top_modules.cycle.set_power.ret_info.label = &&move_backward;

        // call dual.set_power
        current_module = &top_modules.cycle;
        goto *cycle_labels.next_start_labels[0];

    case 0:
    case 1:
    case 2:
        // dual.set_power leftMotor: 0 rightMotor: 0
        top_modules.dual.set_power.kw_leftMotor.param1_f = 0;
        top_modules.dual.set_power.kw_rightMotor.param1_f = 0;
        top_modules.dual.set_power.ret_info.module_context = current_module;
        top_modules.dual.set_power.ret_info.label = &&move_ret5;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual_labels.set_power_start_labels[0];

    move_ret5:
        // return
        current_module =
          ((autonomous_module *)current_module)->move_vars.ret_info
                                                .module_context;
        goto *((autonomous_module *)current_module)->move_vars.ret_info.label;
    }
}
