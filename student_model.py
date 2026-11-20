from pydantic import BaseModel


class Student(BaseModel):
    name: str
    age: int=25
    skill: str


student = Student(
    name="Rahul",
   
    skill="Python"
)

print(student)
print(student.name)
print(student.model_dump())