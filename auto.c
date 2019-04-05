// autonomous.c


struct sub_ret_s {
    void *module_context;
    void *label;
};


struct pos_param_block__f {
    double param1_f;
    double param2_f;
    double param3_f;
    double param4_f;
    double param5_f;
};


struct fn_ret_s {
    struct sub_ret_s sub_ret_info;
    struct pos_param_block__f *ret_params;
};


struct pos_param_block__f_with_return {
    struct sub_ret_s sub_ret_info;
    struct pos_param_block__f params;
};


struct init_sub {
    struct sub_ret_s sub_ret_info;
};


struct start_sub {
    struct sub_ret_s sub_ret_info;
};


struct getPosition_fn {
    struct fn_ret_s fn_ret_info;
    double start;
    double position;
};


struct move_sub {
    struct sub_ret_s sub_ret_info;
    double dist;
    double start;
    unsigned long dlt_mask;
    double tmp1_float;
};


struct autonomous_module {
    struct init_sub       init_vars;
    struct start_sub      start_vars;
    struct getPosition_fn getPosition_vars;
    struct move_sub       move_vars;
};


struct set_power_sub {
    struct sub_ret_s sub_ret_info;
    double leftMotor;
    double rightMotor;
};


struct dual_module {
    struct set_power_sub  set_power_vars;
};


struct report_sub {
    struct sub_ret_s sub_ret_info;
    char *key;
    double value;
};


struct telemetry_module {
    struct report_sub  report_vars;
};


struct next_sub {
    struct sub_ret_s sub_ret_info;
};


struct cycle_module {
    struct next_sub  next_vars;
};


struct {
    struct autonomous_module autonomous;
    struct dual_module dual;
    struct telemetry_module telemetry;
    struct cycle_module cycle;
    //...
} top_modules;


int
main(int argc, char **argv, char **env) {

  void *autonomous__init_start_labels[] = {&&init_start0};
  void *autonomous__start_start_labels[] = {&&start_start0};
  void *autonomous__getPosition_start_labels[] = {
          &&getPosition_start0, &&getPosition_start1};
  void *autonomous__move_start_labels[] = {&&too_few_arguments, &&move_start1};

  void *dual__set_power_start_labels[1];
  void *telemetry__report_start_labels[2];
  void *cycle__next_start_labels[1];

  void *current_module;

  init_start0:
    // set leftMotor.reversed? to: true?
    top_modules.leftMotor.reversed_ = 1;

    // return
    current_module =
      ((struct autonomous_module *)current_module)->init_vars.sub_ret_info.module_context;
    goto *((struct autonomous_module *)current_module)->init_vars.sub_ret_info.label;


  start_start0:
    // move 5ft
    ((struct autonomous_module *)current_module)->move_vars.dist = 60;
    ((struct autonomous_module *)current_module)->move_vars.sub_ret_info.module_context
      = current_module;
    ((struct autonomous_module *)current_module)->move_vars.sub_ret_info.label
      = &&start_ret;

    // call move
    goto *autonomous__move_start_labels[1];
  
  start_ret:
    // return
    current_module =
      ((struct autonomous_module *)current_module)->start_vars.sub_ret_info.module_context;
    goto *((struct autonomous_module *)current_module)->start_vars.sub_ret_info.label;


  getPosition_start0:
    // start = 0
    ((struct autonomous_module *)current_module)->getPosition_vars.start = 0;

  getPosition_start1:
    // set position to:
    // -> (leftMotor.position + rightMotor.position) / 2 * 66.7 - start
    ((struct autonomous_module *)current_module)->getPosition_vars.position = 
      (leftMotor.position + rightMotor.position) / (2 * 66.7) - 
      ((struct autonomous_module *)current_module)->getPosition_vars.start;

    // telemetry.report key: "position" value: position
    top_modules.telemetry.report_vars.key = "position";
    top_modules.telemetry.report_vars.value = 
      ((struct autonomous_module *)current_module)->getPosition_vars.position;
    top_modules.telemetry.report_vars.sub_ret_info.module_context = current_module;
    top_modules.telemetry.report_vars.sub_ret_info.label = &&getPosition_ret;

    // call telemetry.report
    current_module = &top_modules.telemetry;
    goto *telemetry__report_start_labels[1];

  getPosition_ret:
    // return position
    ((struct autonomous_module *)current_module)->getPosition_vars.fn_ret_info
                                                 .ret_params->param1_f =
      ((struct autonomous_module *)current_module)->getPosition_vars.position;
    current_module =
      ((struct autonomous_module *)current_module)->getPosition_vars.fn_ret_info
                                                   .sub_ret_info.module_context;
    goto *((struct autonomous_module *)current_module)->getPosition_vars
                                                       .fn_ret_info
                                                       .sub_ret_info.label;

  move_start1:
    // set start to: {getPosition}
    ((struct autonomous_module *)current_module)->getPosition_vars.fn_ret_info
                                                 .ret_params->param1_f
      = &((struct autonomous_module *)current_module)->move_vars.start;

    ((struct autonomous_module *)current_module)->getPosition_vars.fn_ret_info
                                                 .sub_ret_info.module_context
      = current_module;
    ((struct autonomous_module *)current_module)->getPosition_vars.fn_ret_info
                                                 .sub_ret_info.label
      = &&move_ret1;

    // call getPosition
    goto *autonomous__getPosition_start_labels[0];

  move_ret1:
    // dlt conditions
    ((struct autonomous_module *)current_module)->move_vars.dlt_mask = 0;
    if (((struct autonomous_module *)current_module)->move_vars.dist >= 0) {
        ((struct autonomous_module *)current_module)->move_vars.dlt_mask |= 1;
    }

    // dlt actions
    switch (((struct autonomous_module *)current_module)->move_vars.dlt_mask) {
    case 1:
        goto move_forward;
    case 0:
        goto move_backward;
    }

  move_forward:
    // dlt conditions
    ((struct autonomous_module *)current_module)->move_vars.dlt_mask = 0;
    if (top_modules.isActive) {
        ((struct autonomous_module *)current_module)->move_vars.dlt_mask |= 2;
    }
    if (((struct autonomous_module *)current_module)->move_vars.tmp1_float < 
        ((struct autonomous_module *)current_module)->move_vars.dist
    ) {
        ((struct autonomous_module *)current_module)->move_vars.dlt_mask |= 1;
    }

    // dlt actions
    switch (((struct autonomous_module *)current_module)->move_vars.dlt_mask) {
    case 3:
        // dual.set_power leftMotor: 100% rightMotor: 100%
        top_modules.dual.set_power_vars.leftMotor = 1.0;
        top_modules.dual.set_power_vars.rightMotor = 1.0;
        top_modules.dual.set_power_vars.sub_ret_info.module_context = current_module;
        top_modules.dual.set_power_vars.sub_ret_info.label = &&move_ret2;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual__set_power_start_labels[0];

    move_ret2:
        // goto cycle.next returning_to: forward
        top_modules.cycle.next_vars.sub_ret_info.module_context = current_module;
        top_modules.cycle.next_vars.sub_ret_info.label = &&move_forward;

        // call dual.set_power
        current_module = &top_modules.cycle;
        goto *cycle__next_start_labels[0];

    case 0:
    case 1:
    case 2:
        // dual.set_power leftMotor: 0 rightMotor: 0
        top_modules.dual.set_power_vars.leftMotor = 0;
        top_modules.dual.set_power_vars.rightMotor = 0;
        top_modules.dual.set_power_vars.sub_ret_info.module_context = current_module;
        top_modules.dual.set_power_vars.sub_ret_info.label = &&move_ret3;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual__set_power_start_labels[0];

    move_ret3:
        // return
        current_module =
          ((struct autonomous_module *)current_module)->move_vars.sub_ret_info
                                                       .module_context;
        goto *((struct autonomous_module *)current_module)->move_vars.sub_ret_info.label;
    }

  move_backward:
    // dlt conditions
    ((struct autonomous_module *)current_module)->move_vars.dlt_mask = 0;
    if (top_modules.isActive) {
        ((struct autonomous_module *)current_module)->move_vars.dlt_mask |= 2;
    }
    if (((struct autonomous_module *)current_module)->move_vars.tmp1_float >
        ((struct autonomous_module *)current_module)->move_vars.dist
    ) {
        ((struct autonomous_module *)current_module)->move_vars.dlt_mask |= 1;
    }

    // dlt actions
    switch (((struct autonomous_module *)current_module)->move_vars.dlt_mask) {
    case 3:
        // dual.set_power leftMotor: -100% rightMotor: -100%
        top_modules.dual.set_power_vars.leftMotor = -1.0;
        top_modules.dual.set_power_vars.rightMotor = -1.0;
        top_modules.dual.set_power_vars.sub_ret_info.module_context = current_module;
        top_modules.dual.set_power_vars.sub_ret_info.label = &&move_ret4;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual__set_power_start_labels[0];

    move_ret4:
        // goto cycle.next returning_to: forward
        top_modules.cycle.next_vars.sub_ret_info.module_context = current_module;
        top_modules.cycle.next_vars.sub_ret_info.label = &&move_backward;

        // call dual.set_power
        current_module = &top_modules.cycle;
        goto *cycle__next_start_labels[0];

    case 0:
    case 1:
    case 2:
        // dual.set_power leftMotor: 0 rightMotor: 0
        top_modules.dual.set_power_vars.leftMotor = 0;
        top_modules.dual.set_power_vars.rightMotor = 0;
        top_modules.dual.set_power_vars.sub_ret_info.module_context = current_module;
        top_modules.dual.set_power_vars.sub_ret_info.label = &&move_ret5;

        // call dual.set_power
        current_module = &top_modules.dual;
        goto *dual__set_power_start_labels[0];

    move_ret5:
        // return
        current_module =
          ((struct autonomous_module *)current_module)->move_vars.sub_ret_info
                                                .module_context;
        goto *((struct autonomous_module *)current_module)->move_vars.sub_ret_info.label;
    }
}
