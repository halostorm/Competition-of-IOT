INCLUDE = $(shell pkg-config --cflags opencv)
LIBS = $(shell pkg-config --libs opencv)
SOURCES = tracker.cpp
# 目标文件
OBJECTS = $(SOURCES:.cpp=.o)
# 可执行文件
TARGET = tracker
$(TARGET):$(OBJECTS)
	g++ -o $(TARGET) $(OBJECTS) -I -lm $(INCLUDE) $(LIBS)
$(OBJECTS):$(SOURCES)
	g++ -c $(SOURCES)
clean:
	rm $(OBJECTS) $(TARGET)
%.o:%.cpp
	g++ -I $(INCLUDE) -o $@ -c $<
