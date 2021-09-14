#include <helper/Program.hpp>

int main(int argument_count, char** argument_variables){
    auto program = Program(argument_count, argument_variables);
    return program.run();
}